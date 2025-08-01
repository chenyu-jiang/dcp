import argparse
import datetime
import json
import os
from typing import List
import random

import torch
import torch.distributed as dist
import torch.nn.functional as F
import tqdm
from flash_attn import flash_attn_varlen_kvpacked_func
from dcp_flash_attn import (
    flash_attn_varlen_kvpacked_func as dcp_flash_attn_varlen_kvpacked_func,
)
from utils import (
    benchmark_f,
    extract_local_ring,
    extract_local_zigzag,
    get_lse_diff,
    init_env,
    print_0,
    print_all,
    unpack_qkv_before_attn,
)

from dcp.core.common import get_default_execution_context
from dcp.core.instructions import ExecutionPlan
from dcp.runtime.flash_attention import AttentionExecutor, get_executor
from dcp.runtime.flash_attention.executor import AttentionExecutor
from dcp.runtime.flash_attention.interface import (
    get_dataloader_for_benchmark,
    get_local_q_kv,
)
from dcp.runtime.flash_attention.utils import (
    reconstruct_attention_forward_output_for_test,
    lambda_mask_fn,
    shared_question_mask_fn,
    causal_blockwise_mask_fn,
)


def get_mask_fn(mask_type: str):
    if mask_type == "causal":
        return None
    elif mask_type == "lambda":
        return lambda_mask_fn
    elif mask_type == "shared_question":
        return shared_question_mask_fn
    elif mask_type == "causal_blockwise":
        return causal_blockwise_mask_fn
    else:
        raise ValueError(f"Unsupported mask type: {mask_type}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-nd", "--n-devices-per-node", type=int, default=8)
    parser.add_argument("-me", "--mem-epsilon", type=float, default=0.2)
    parser.add_argument("-ce", "--comp-epsilon", type=float, default=0.1)
    parser.add_argument(
        "--inter-node-comp-imbalance-factor", type=float, default=4.0
    )
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--block-size", type=int, default=512)
    parser.add_argument("--head-block-size", type=int, default=1)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--n-query-groups", type=int, default=2)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument(
        "-b",
        "--benchmarks",
        type=str,
        required=True,
        choices=["local", "te", "lt", "zigzag", "ring", "dcp"],
    )
    parser.add_argument(
        "-bt",
        "--benchmark-type",
        type=str,
        default="forward",
        choices=["forward", "backward"],
    )
    parser.add_argument(
        "-m",
        "--mask",
        type=str,
        default="causal",
        choices=["causal", "lambda", "shared_question", "causal_blockwise"],
    )
    parser.add_argument(
        "--seqlens-config",
        type=str,
        help="Path to a config file containing a list of sequence lengths to "
        "be used for benchmarking",
    )
    parser.add_argument(
        "--seqlens",
        type=str,
        help="Comma separated list of sequence lengths, overrides default",
    )
    parser.add_argument(
        "--profile", action="store_true", help="Profile with Nsight Systems"
    )
    parser.add_argument(
        "--profiler-impl",
        type=str,
        default="nsys",
        choices=["nsys", "torch"],
        help="Profiler implementation to use",
    )
    parser.add_argument(
        "--profiler-out-dir",
        type=str,
        default="./",
        help="Output dir for the torch profiler trace",
    )
    parser.add_argument(
        "-c",
        "--check-correctness",
        action="store_true",
        help="Check correctness",
    )
    parser.add_argument(
        "-cg",
        "--use-cudagraph",
        action="store_true",
        help="Use CUDAGraph for benchmarking",
    )
    parser.add_argument(
        "-t",
        "--te-cp-comm-type",
        type=str,
        default="a2a+p2p",
        choices=["p2p", "a2a", "a2a+p2p"],
    )
    parser.add_argument(
        "--p2p-a2a-order",
        type=str,
        default="AP",
        choices=["AP", "PA"],
        help="AP: a2a have consecutive ranks; PA: reverse.",
    )
    parser.add_argument(
        "--a2a-degree",
        type=int,
        default=-1,
        help="Degree of head parallelism. -1 means parallelize all heads.",
    )
    parser.add_argument(
        "--dp-degree",
        type=int,
        default=1,
        help="Degree of data parallelism.",
    )
    parser.add_argument(
        "--lt-window-size",
        type=int,
        default=-1,
        help="LoongTrain window size double ring attention",
    )
    parser.add_argument(
        "--lt-interleaved",
        action="store_true",
        help="LoongTrain interleaved GPU placement for double ring attention",
    )
    parser.add_argument(
        "--silent",
        action="store_true",
        help="Suppress output like tqdm progress bar",
    )
    parser.add_argument(
        "--num-preprocessing-workers",
        type=int,
        default=16,
        help="Number of workers for plan generation on each node",
    )
    parser.add_argument(
        "--use-block-size-heuristic",
        action="store_true",
        help="Use block size heuristic for plan generation instead of fixed",
    )
    parser.add_argument("-ns", "--n-samples", type=int, default=500)
    args = parser.parse_args()

    args.dropout_p = 0.0  # we don't support dropout for now

    if args.dtype == "bfloat16":
        args.dtype = torch.bfloat16
    elif args.dtype == "float16":
        args.dtype = torch.float16
    else:
        raise ValueError("Unsupported dtype {}".format(args.dtype))

    if args.check_correctness:
        args.profile = False
        assert (
            args.benchmarks != "local"
        ), "No need to check correctness for local"

    if args.seqlens is None:
        args.seqlens = [1024, 2048, 4096, 8192, 1024, 3072, 1024, 1024]
    else:
        args.seqlens = list(map(int, args.seqlens.split(",")))

    if not args.seqlens_config:
        # single sequence, set num processing workers to 1
        args.num_preprocessing_workers = 1

    if args.mask != "causal":
        assert args.benchmarks in [
            "local",
            "dcp",
            "te",
        ], "Only local, dcp and te support non-causal masks"
    return args


def exec_dcp_distributed_attn(
    executor: AttentionExecutor, exec_plan: ExecutionPlan
):
    return executor.execute(exec_plan)


def prepare_local_flash_attn(seqlens, q, kv):
    d = q.shape[-1]
    softmax_scale = d ** (-0.5)
    cu_seqlens = F.pad(
        torch.cumsum(torch.tensor(seqlens, dtype=torch.int32), 0), (1, 0)
    ).to(torch.int32)
    total_seqlens = sum(seqlens)
    max_seqlens = max(seqlens)
    return softmax_scale, cu_seqlens, total_seqlens, max_seqlens


def prepare_local_dcp_flash_attn(config, seqlens, q, kv):
    assert config.mask != "causal"
    d = q.shape[-1]
    softmax_scale = d ** (-0.5)
    cu_seqlens = F.pad(
        torch.cumsum(torch.tensor(seqlens, dtype=torch.int32), 0), (1, 0)
    ).to(torch.int32)
    total_seqlens = sum(seqlens)
    max_seqlens = max(seqlens)
    mask_fn = get_mask_fn(config.mask)
    attn_mask = mask_fn(seqlens, seqlens, torch.int32)
    if attn_mask.dim() == 3:
        attn_mask = attn_mask.permute(1, 2, 0).contiguous()
    else:
        attn_mask = attn_mask.transpose(1, 0).contiguous()
    return softmax_scale, cu_seqlens, total_seqlens, max_seqlens, attn_mask


def exec_local_flash_attn(
    args,
    q,
    kv,
    cu_seqlens,
    total_seqlens,
    max_seqlens,
    softmax_scale,
):
    assert args.mask == "causal"
    out, lse, _ = flash_attn_varlen_kvpacked_func(
        q,
        kv,
        cu_seqlens,
        cu_seqlens,
        max_seqlens,
        max_seqlens,
        args.dropout_p,
        softmax_scale,
        causal=True,
        deterministic=args.deterministic,
        return_attn_probs=True,
    )
    return out, lse


def exec_local_dcp_flash_attn(
    args,
    q,
    kv,
    cu_seqlens,
    total_seqlens,
    max_seqlens,
    softmax_scale,
    attn_mask,
):
    out, lse, _ = dcp_flash_attn_varlen_kvpacked_func(
        q,
        kv,
        cu_seqlens,
        cu_seqlens,
        total_seqlens,
        total_seqlens,
        max_seqlens,
        max_seqlens,
        args.dropout_p,
        softmax_scale,
        return_attn_probs=True,
        attn_range=attn_mask,
        force_split_kv=True,
    )
    return out, lse


def generate_input(args, batch_seqlens: List[List[int]]):
    torch.manual_seed(args.seed)
    if args.benchmarks == "dcp":
        # create dataloader
        batch_indices = []
        flattened_seqlens = []
        for seqlens in batch_seqlens:
            batch_indices.append(
                list(
                    range(
                        len(flattened_seqlens),
                        len(flattened_seqlens) + len(seqlens),
                    )
                )
            )
            flattened_seqlens.extend(seqlens)
        dataloader = get_dataloader_for_benchmark(
            flattened_seqlens,
            batch_indices,
            args.mask,
            args.n_heads,
            args.n_query_groups,
            args.head_dim,
            args.n_devices_per_node,
            args.block_size,
            args.head_block_size,
            args.dtype,
            args.mem_epsilon,
            args.comp_epsilon,
            args.inter_node_comp_imbalance_factor,
            args.num_preprocessing_workers,
            args.use_block_size_heuristic,
            mask_fn=get_mask_fn(args.mask),
        )
        for batch_idx, (
            _,
            fw_exec_plan,
            bw_exec_plan,
            fw_workload,
            bw_workload,
            _,
        ) in enumerate(dataloader):
            raw_seqlens = batch_seqlens[batch_idx]
            q = torch.randn(
                sum(raw_seqlens),
                args.n_heads,
                args.head_dim,
                dtype=args.dtype,
            )
            kv = torch.randn(
                sum(raw_seqlens),
                2,
                args.n_query_groups,
                args.head_dim,
                dtype=args.dtype,
            )
            dout = torch.randn(
                sum(raw_seqlens),
                args.n_heads,
                args.head_dim,
                dtype=args.dtype,
            )
            dist.broadcast(
                q,
                src=dist.get_global_rank(args.cp_group, 0),
                group=args.cp_group,
            )
            dist.broadcast(
                kv,
                src=dist.get_global_rank(args.cp_group, 0),
                group=args.cp_group,
            )
            dist.broadcast(
                dout,
                src=dist.get_global_rank(args.cp_group, 0),
                group=args.cp_group,
            )
            local_rank = int(os.environ["LOCAL_RANK"])
            local_q, local_kv, local_dout = get_local_q_kv(
                q,
                kv,
                batch_seqlens[batch_idx],
                fw_exec_plan,
                fw_workload,
                args.cp_rank // args.n_devices_per_node,
                local_rank,
                args.head_block_size,
                bw_workload,
                dout=dout,
            )
            local_q = local_q.contiguous()
            local_kv = local_kv.contiguous()
            local_dout = local_dout.contiguous()
            yield q, kv, dout, batch_seqlens[batch_idx], {
                "local_q": local_q,
                "local_kv": local_kv,
                "local_dout": local_dout,
                "fw_exec_plan": fw_exec_plan,
                "bw_exec_plan": bw_exec_plan,
                "fw_workload": fw_workload,
                "bw_workload": bw_workload,
            }
    else:
        padded_batch_seqlens = []
        for batch_idx, seqlens in enumerate(batch_seqlens):
            # if args.benchmarks == "te" and batch_idx == 105:
            #     # some how it crashes on this batch
            #     continue
            if args.dp_degree > 1:
                assert (
                    len(seqlens) >= args.dp_degree
                ), "Not enough sequences for dp_degree {}.".format(
                    args.dp_degree
                )
                # split seqlens into dp_degree chunks
                total_tokens = sum(seqlens)
                per_dpg_seqlens = [[] for _ in range(args.dp_degree)]
                target_tokens = total_tokens // args.dp_degree
                # greedy
                for seqlen in seqlens:
                    # find the smallest one
                    min_idx = min(
                        range(args.dp_degree),
                        key=lambda i: sum(per_dpg_seqlens[i]),
                    )
                    per_dpg_seqlens[min_idx].append(seqlen)
                # get local data
                seqlens = per_dpg_seqlens[args.dp_rank]
            # round up to the nearest multiple of 2 * world_size
            seqlens = [
                (seqlen + 2 * args.cp_world_size - 1)
                // (2 * args.cp_world_size)
                * (2 * args.cp_world_size)
                for seqlen in seqlens
            ]
            padded_batch_seqlens.append(seqlens)
        if args.benchmarks == "te" and args.mask != "causal":
            from baselines.transformer_engine.transformer_engine import (
                generate_masks_for_te,
            )

            mask_generator = iter(
                generate_masks_for_te(args, padded_batch_seqlens)
            )
        for seqlens in padded_batch_seqlens:
            q = torch.randn(
                sum(seqlens), args.n_heads, args.head_dim, dtype=args.dtype
            )
            kv = torch.randn(
                sum(seqlens),
                2,
                args.n_query_groups,
                args.head_dim,
                dtype=args.dtype,
            )
            dout = torch.randn(
                sum(seqlens), args.n_heads, args.head_dim, dtype=args.dtype
            )
            if args.benchmark_type == "backward":
                q = q.requires_grad_()
                kv = kv.requires_grad_()
                dout = dout.requires_grad_()
            dist.broadcast(
                q,
                src=dist.get_global_rank(args.cp_group, 0),
                group=args.cp_group,
            )
            dist.broadcast(
                kv,
                src=dist.get_global_rank(args.cp_group, 0),
                group=args.cp_group,
            )
            dist.broadcast(
                dout,
                src=dist.get_global_rank(args.cp_group, 0),
                group=args.cp_group,
            )
            extra_args = {}
            if args.benchmarks == "te" and args.mask != "causal":
                # generate masks for te
                per_step_attn_masks = next(mask_generator)
                per_step_attn_masks = [t.cuda() for t in per_step_attn_masks]
                extra_args["te_per_step_attn_masks"] = per_step_attn_masks
            yield q, kv, dout, seqlens, extra_args


def prepare_and_gen_args(args, inputs_per_batch):
    for q, kv, dout, seqlens, extras in inputs_per_batch:
        args_dict = {}
        args_dict["seqlens"] = seqlens
        args_dict["q_shape"] = q.shape
        args_dict["kv_shape"] = kv.shape
        args_dict["q_dtype"] = q.dtype
        # Prepare
        if args.benchmarks == "local" or args.check_correctness:
            args_dict["local_dout"] = dout
            if args.mask == "causal":
                softmax_scale, cu_seqlens, total_seqlens, max_seqlens = (
                    prepare_local_flash_attn(seqlens, q, kv)
                )

                args_dict["local"] = (
                    (
                        args,
                        q,
                        kv,
                        cu_seqlens,
                        total_seqlens,
                        max_seqlens,
                        softmax_scale,
                    ),
                    {},
                )
            else:
                (
                    softmax_scale,
                    cu_seqlens,
                    total_seqlens,
                    max_seqlens,
                    attn_mask,
                ) = prepare_local_dcp_flash_attn(args, seqlens, q, kv)

                args_dict["local"] = (
                    (
                        args,
                        q,
                        kv,
                        cu_seqlens,
                        total_seqlens,
                        max_seqlens,
                        softmax_scale,
                        attn_mask,
                    ),
                    {},
                )

        if args.benchmarks == "zigzag":
            from baselines.ring_flash_attn.ring_flash_attn import (
                prepare_zigzag_ring_attn,
            )

            (
                zigzag_local_q,
                zigzag_local_kv,
                zigzag_local_dout,
                zigzag_local_cu_seqlens,
                zigzag_local_max_seqlens,
            ) = prepare_zigzag_ring_attn(args, seqlens, q, kv, dout)
            softmax_scale = q.shape[-1] ** (-0.5)

            if args.benchmark_type == "backward":
                zigzag_local_q.requires_grad_(True)
                zigzag_local_kv.requires_grad_(True)
                zigzag_local_dout.requires_grad_(True)

            args_dict["zigzag"] = (
                (
                    zigzag_local_q,
                    zigzag_local_kv,
                    zigzag_local_cu_seqlens,
                    zigzag_local_max_seqlens,
                    args.dropout_p,
                    softmax_scale,
                ),
                {
                    "causal": True,
                    "deterministic": args.deterministic,
                    "return_attn_probs": True,
                    "group": args.cp_group,
                },
            )
            args_dict["zigzag_local_dout"] = zigzag_local_dout

        if args.benchmarks == "ring":
            from baselines.ring_flash_attn.ring_flash_attn import (
                prepare_ring_attn,
            )

            (
                ring_local_q,
                ring_local_kv,
                ring_local_dout,
                ring_local_cu_seqlens,
                ring_local_max_seqlens,
            ) = prepare_ring_attn(args, seqlens, q, kv, dout)
            softmax_scale = q.shape[-1] ** (-0.5)

            if args.benchmark_type == "backward":
                ring_local_q.requires_grad_(True)
                ring_local_kv.requires_grad_(True)
                ring_local_dout.requires_grad_(True)

            args_dict["ring"] = (
                (
                    ring_local_q,
                    ring_local_kv,
                    ring_local_cu_seqlens,
                    ring_local_max_seqlens,
                    args.dropout_p,
                    softmax_scale,
                ),
                {
                    "causal": True,
                    "deterministic": args.deterministic,
                    "return_attn_probs": True,
                    "group": args.cp_group,
                },
            )
            args_dict["ring_local_dout"] = ring_local_dout

        if args.benchmarks == "te":
            from baselines.transformer_engine.transformer_engine import (
                prepare_te_attn,
            )

            (
                te_attn_mod,
                te_local_q,
                te_local_kv,
                te_local_dout,
                te_cu_seqlens,
                te_cu_seqlens_cpu,
                te_max_seqlen,
            ) = prepare_te_attn(args, seqlens, q, kv, dout)

            if args.mask != "causal":
                per_step_attn_masks = extras["te_per_step_attn_masks"]
            else:
                per_step_attn_masks = None

            te_q = te_local_q
            te_k = te_local_kv[:, 0]
            te_v = te_local_kv[:, 1]
            if args.benchmark_type == "backward":
                te_q.requires_grad_(True)
                te_k.requires_grad_(True)
                te_v.requires_grad_(True)
                te_local_dout.requires_grad_(True)
            args_dict["te"] = (
                te_attn_mod,
                (te_q, te_k, te_v),
                {
                    "core_attention_bias_type": "no_bias",
                    "cu_seqlens_q": te_cu_seqlens,
                    "cu_seqlens_kv": te_cu_seqlens,
                    "max_seqlen_q": te_max_seqlen,
                    "max_seqlen_kv": te_max_seqlen,
                    "attn_ranges_per_step": per_step_attn_masks,
                    "cu_seqlens_q_cpu": te_cu_seqlens_cpu,
                    "cu_seqlens_kv_cpu": te_cu_seqlens_cpu,
                },
            )
            args_dict["te_local_dout"] = te_local_dout.reshape(
                te_local_dout.shape[0], -1
            )

        if args.benchmarks == "lt":
            from baselines.loong_train.loong_train import prepare_lt_attn

            (
                lt_local_q,
                lt_local_kv,
                lt_local_dout,
                lt_attn_mod,
                lt_kwargs,
                get_local_chunk_fn,
                zero_out_local_padding_fn,
            ) = prepare_lt_attn(args, seqlens, q, kv, dout)

            args_dict["lt"] = (
                lt_attn_mod,
                (lt_local_q, lt_local_kv),
                lt_kwargs,
            )
            args_dict["lt_get_local_chunk_fn"] = get_local_chunk_fn
            args_dict["lt_zero_out_local_padding_fn"] = (
                zero_out_local_padding_fn
            )
            args_dict["lt_local_dout"] = lt_local_dout

        if args.benchmarks == "dcp":
            n_nodes = dist.get_world_size() // args.n_devices_per_node
            exec_context = get_default_execution_context(
                dist.get_world_size(),
                n_nodes,
            )
            executor = get_executor(
                args.head_block_size, args.head_dim, exec_context
            )

            local_q = extras["local_q"]
            local_kv = extras["local_kv"]
            local_dout = extras["local_dout"]
            fw_exec_plan = extras["fw_exec_plan"]
            bw_exec_plan = extras["bw_exec_plan"]
            fw_workload = extras["fw_workload"]
            bw_workload = extras["bw_workload"]

            args_dict["dcp"] = (executor, fw_exec_plan, local_q, local_kv)
            args_dict["dcp_fw_workload"] = fw_workload
            args_dict["dcp_bw_exec_plan"] = bw_exec_plan
            args_dict["dcp_bw_workload"] = bw_workload
            args_dict["dcp_local_dout"] = local_dout

        yield args_dict


def _print_diff(
    test_name: str,
    tensor_name: str,
    diff_mean: float,
    diff_max: float,
    print_on_all_devs=False,
):
    if print_on_all_devs:
        print_fn = print_all
    else:
        print_fn = print_0
    print_fn(
        f"{test_name}: {tensor_name} diff mean {diff_mean:.4f}, {tensor_name} diff max {diff_max:.4f}"
    )


def _recursive_to_cuda(x):
    if isinstance(x, dict):
        res = {}
        for k, v in x.items():
            if isinstance(k, str) and "cpu" in k:
                res[k] = v
            else:
                res[k] = _recursive_to_cuda(v)
        return res
    elif isinstance(x, list):
        return [_recursive_to_cuda(v) for v in x]
    elif isinstance(x, tuple):
        return tuple(_recursive_to_cuda(v) for v in x)
    elif isinstance(x, torch.Tensor):
        return x.cuda()
    else:
        return x


def _bench_local_attn(args, args_dict):
    args_dict["local"] = _recursive_to_cuda(args_dict["local"])

    if args.mask == "causal":

        def local_flash_attn_func():
            return exec_local_flash_attn(*args_dict["local"][0])

    else:

        def local_flash_attn_func():
            return exec_local_dcp_flash_attn(*args_dict["local"][0])

    if args.benchmark_type == "forward":

        _, local_flash_time = benchmark_f(
            local_flash_attn_func,
            args.warmup,
            args.iters,
            args.profile,
            args.profiler_impl,
            args.profiler_out_dir,
            use_cudagraph=args.use_cudagraph,
        )
    else:
        local_out, local_lse = local_flash_attn_func()
        local_q = args_dict["local"][0][1]
        local_kv = args_dict["local"][0][2]
        local_dout = args_dict["local_dout"]

        def local_flash_attn_bw_func():
            return torch.autograd.grad(
                local_out, (local_q, local_kv), local_dout, retain_graph=True
            )

        _, local_flash_time = benchmark_f(
            local_flash_attn_bw_func,
            args.warmup,
            args.iters,
            args.profile,
            args.profiler_impl,
            args.profiler_out_dir,
            use_cudagraph=args.use_cudagraph,
        )
    return local_flash_time


def _bench_zigzag(args, args_dict):
    from baselines.ring_flash_attn.ring_flash_attn import (
        zigzag_ring_flash_attn_varlen_kvpacked_func,
    )

    args_dict["zigzag"] = _recursive_to_cuda(args_dict["zigzag"])

    def zigzag_ring_attn_func():
        return zigzag_ring_flash_attn_varlen_kvpacked_func(
            *args_dict["zigzag"][0],
            **args_dict["zigzag"][1],
        )

    if args.benchmark_type == "forward":
        _, zigzag_ring_time = benchmark_f(
            zigzag_ring_attn_func,
            args.warmup,
            args.iters,
            args.profile,
            args.profiler_impl,
            args.profiler_out_dir,
            use_cudagraph=args.use_cudagraph,
        )
    else:
        out, _, _ = zigzag_ring_attn_func()

        local_q = args_dict["zigzag"][0][0]
        local_kv = args_dict["zigzag"][0][1]
        local_dout = args_dict["zigzag_local_dout"]

        def zigzag_ring_attn_bw_func():
            return torch.autograd.grad(
                out, (local_q, local_kv), local_dout, retain_graph=True
            )

        _, zigzag_ring_time = benchmark_f(
            zigzag_ring_attn_bw_func,
            args.warmup,
            args.iters,
            args.profile,
            args.profiler_impl,
            args.profiler_out_dir,
            use_cudagraph=args.use_cudagraph,
        )
    return zigzag_ring_time


def _bench_ring(args, args_dict):
    from baselines.ring_flash_attn.ring_flash_attn import (
        ring_flash_attn_varlen_kvpacked_func,
    )

    args_dict["ring"] = _recursive_to_cuda(args_dict["ring"])

    def ring_attn_func():
        return ring_flash_attn_varlen_kvpacked_func(
            *args_dict["ring"][0], **args_dict["ring"][1]
        )

    if args.benchmark_type == "forward":
        _, ring_time = benchmark_f(
            ring_attn_func,
            args.warmup,
            args.iters,
            args.profile,
            args.profiler_impl,
            args.profiler_out_dir,
            use_cudagraph=args.use_cudagraph,
        )
    else:
        out, _, _ = ring_attn_func()

        local_q = args_dict["ring"][0][0]
        local_kv = args_dict["ring"][0][1]
        local_dout = args_dict["ring_local_dout"]

        def ring_attn_bw_func():
            return torch.autograd.grad(
                out, (local_q, local_kv), local_dout, retain_graph=True
            )

        _, ring_time = benchmark_f(
            ring_attn_bw_func,
            args.warmup,
            args.iters,
            args.profile,
            args.profiler_impl,
            args.profiler_out_dir,
            use_cudagraph=args.use_cudagraph,
        )
    return ring_time


def _bench_te(args, args_dict):
    args_dict["te"] = _recursive_to_cuda(args_dict["te"])

    def te_attn_func():
        r = args_dict["te"][0](*args_dict["te"][1], **args_dict["te"][2])
        return r

    if args.benchmark_type == "forward":
        _, te_time = benchmark_f(
            te_attn_func,
            args.warmup,
            args.iters,
            args.profile,
            args.profiler_impl,
            args.profiler_out_dir,
            use_cudagraph=args.use_cudagraph,
        )
    else:
        out = te_attn_func()
        local_q = args_dict["te"][1][0]
        local_k = args_dict["te"][1][1]
        local_v = args_dict["te"][1][2]

        local_dout = args_dict["te_local_dout"]

        def te_attn_bw_func():
            return torch.autograd.grad(
                out, (local_q, local_k, local_v), local_dout, retain_graph=True
            )

        _, te_time = benchmark_f(
            te_attn_bw_func,
            args.warmup,
            args.iters,
            args.profile,
            args.profiler_impl,
            args.profiler_out_dir,
            use_cudagraph=args.use_cudagraph,
        )
    return te_time


def _bench_lt(args, args_dict):
    args_dict["lt"] = _recursive_to_cuda(args_dict["lt"])

    def lt_attn_func():
        return args_dict["lt"][0](*args_dict["lt"][1], **args_dict["lt"][2])

    if args.benchmark_type == "forward":
        _, lt_time = benchmark_f(
            lt_attn_func,
            args.warmup,
            args.iters,
            args.profile,
            args.profiler_impl,
            args.profiler_out_dir,
            use_cudagraph=args.use_cudagraph,
        )
    else:
        out = lt_attn_func()
        local_q = args_dict["lt"][1][0]
        local_kv = args_dict["lt"][1][1]
        local_dout = args_dict["lt_local_dout"]
        local_dout = local_dout.reshape(out.shape)

        def lt_attn_bw_func():
            return torch.autograd.grad(
                out, (local_q, local_kv), local_dout, retain_graph=True
            )

        _, lt_time = benchmark_f(
            lt_attn_bw_func,
            args.warmup,
            args.iters,
            args.profile,
            args.profiler_impl,
            args.profiler_out_dir,
            use_cudagraph=args.use_cudagraph,
        )
    return lt_time


def _bench_dcp(args, args_dict):
    args_dict["dcp"] = _recursive_to_cuda(args_dict["dcp"])
    executor, fw_exec_plan, local_q, local_kv = args_dict["dcp"]
    executor.fw_exec_plan = fw_exec_plan
    executor.prepare(fw_exec_plan)
    executor.init_forward_input(local_q, local_kv)

    def dcp_attn_func():
        r = exec_dcp_distributed_attn(executor, fw_exec_plan)
        return r

    if args.benchmark_type == "forward":
        _, dcp_time = benchmark_f(
            dcp_attn_func,
            args.warmup,
            args.iters,
            args.profile,
            args.profiler_impl,
            args.profiler_out_dir,
            use_cudagraph=args.use_cudagraph,
        )
    else:
        local_out, local_lse = dcp_attn_func()
        # (executor, fw_exec_plan, local_q, local_kv)
        bw_exec_plan = args_dict["dcp_bw_exec_plan"]
        local_dout = args_dict["dcp_local_dout"]
        # bw_workload = args_dict["dcp_bw_workload"]
        executor.bw_exec_plan = bw_exec_plan
        executor.switch_to_backward()
        executor.set_local_out_lse(local_out, local_lse)
        executor.init_forward_input(local_q, local_kv)
        executor.init_backward_input(local_dout)

        def dcp_attn_bw_func():
            return executor.execute(bw_exec_plan, is_forward=False)

        _, dcp_time = benchmark_f(
            dcp_attn_bw_func,
            args.warmup,
            args.iters,
            args.profile,
            args.profiler_impl,
            args.profiler_out_dir,
            use_cudagraph=args.use_cudagraph,
        )
    return dcp_time


if __name__ == "__main__":
    args = parse_args()

    dist.init_process_group(
        backend="nccl",
        timeout=datetime.timedelta(minutes=60),
        pg_options=dist.ProcessGroupNCCL.Options(is_high_priority_stream=True),
    )
    init_env()

    if args.n_devices_per_node > dist.get_world_size():
        args.n_devices_per_node = dist.get_world_size()
        if dist.get_rank() == 0:
            print(f"Setting n_devices_per_node to {args.n_devices_per_node}.")

    if args.benchmarks == "dcp":
        assert args.dp_degree == 1, "Data parallelism is a subset of dcp"

    # create sub groups for devices in each dp group
    n_devices_total = dist.get_world_size()
    assert n_devices_total % args.dp_degree == 0
    cp_degree = n_devices_total // args.dp_degree
    cp_group_ranks = [
        list(range(i, i + cp_degree))
        for i in range(0, dist.get_world_size(), cp_degree)
    ]
    dp_group_ranks = [
        [i * cp_degree + j for i in range(args.dp_degree)]
        for j in range(cp_degree)
    ]
    cp_groups = [
        dist.new_group(ranks, backend="nccl") for ranks in cp_group_ranks
    ]
    dp_groups = [
        dist.new_group(ranks, backend="nccl") for ranks in dp_group_ranks
    ]
    cp_rank = dist.get_rank() % cp_degree
    dp_rank = dist.get_rank() // cp_degree
    args.cp_rank = cp_rank
    args.dp_rank = dp_rank
    args.all_cp_groups = cp_groups
    args.all_dp_groups = dp_groups
    args.cp_group = cp_groups[dp_rank]
    args.dp_group = dp_groups[cp_rank]
    args.cp_world_size = cp_degree
    args.curr_cpg_ranks = cp_group_ranks[dp_rank]
    args.curr_dpg_ranks = dp_group_ranks[cp_rank]

    if args.seqlens_config is not None:
        with open(args.seqlens_config, "r") as f:
            seqlens_config = json.load(f)
            batch_seqlens = seqlens_config["batches"]
            max_dp_degree = seqlens_config["max_dp_degree"]
            if args.dp_degree > max_dp_degree:
                raise ValueError(
                    f"dp_degree {args.dp_degree} is greater than max_dp_degree {max_dp_degree}"
                )
        # sample args.n_samples batches
        rng = random.Random(args.seed)
        batch_seqlens = rng.sample(
            batch_seqlens, min(args.n_samples, len(batch_seqlens))
        )
        n_batches = len(batch_seqlens)
    else:
        batch_seqlens = [args.seqlens]
        n_batches = 1

    inputs_per_batch = generate_input(args, batch_seqlens)
    args_dicts = prepare_and_gen_args(args, inputs_per_batch)

    # f_iter_time = open("./dcp_iteration_times.txt", "w")
    # f_iter_count = 0

    if args.check_correctness:
        assert n_batches == 1

        args_dicts = list(args_dicts)

        args_dicts[0]["local"] = _recursive_to_cuda(args_dicts[0]["local"])
        if args.mask == "causal":
            flash_attn_out, flash_attn_lse = exec_local_flash_attn(
                *args_dicts[0]["local"][0]
            )
        else:
            flash_attn_out, flash_attn_lse = exec_local_dcp_flash_attn(
                *args_dicts[0]["local"][0]
            )
        world_size = dist.get_world_size(group=args.cp_group)
        cu_seqlens = F.pad(
            torch.cumsum(
                torch.tensor(args_dicts[0]["seqlens"], dtype=torch.int32), 0
            ),
            (1, 0),
        ).to(torch.int32)
        local_cu_seqlens = cu_seqlens // world_size
        if args.benchmarks == "zigzag":
            from baselines.ring_flash_attn.ring_flash_attn import (
                zigzag_ring_flash_attn_varlen_kvpacked_func,
            )

            print_0("Checking correctness for zigzag ring attn")

            args_dicts[0]["zigzag"] = _recursive_to_cuda(
                args_dicts[0]["zigzag"]
            )
            zigzag_ring_out, zigzag_ring_lse, _ = (
                zigzag_ring_flash_attn_varlen_kvpacked_func(
                    *args_dicts[0]["zigzag"][0], **args_dicts[0]["zigzag"][1]
                )
            )

            flash_attn_out_local = extract_local_zigzag(
                flash_attn_out,
                cu_seqlens,
                dist.get_rank(group=args.cp_group),
                dist.get_world_size(group=args.cp_group),
            )

            out_diff = (flash_attn_out_local - zigzag_ring_out).abs()
            out_diff_mean = out_diff.mean().item()
            out_diff_max = out_diff.max().item()
            lse_diff_mean, lse_diff_max = get_lse_diff(
                flash_attn_lse,
                zigzag_ring_lse,
                cu_seqlens,
                "zigzag",
                args.cp_group,
            )

            print_0("==" * 20)
            _print_diff(
                "Zigzag Ring Attn",
                "out",
                out_diff_mean,
                out_diff_max,
                print_on_all_devs=True,
            )
            _print_diff(
                "Zigzag Ring Attn",
                "lse",
                lse_diff_mean,
                lse_diff_max,
                print_on_all_devs=True,
            )

            torch.cuda.empty_cache()
        if args.benchmarks == "ring":
            from baselines.ring_flash_attn.ring_flash_attn import (
                ring_flash_attn_varlen_kvpacked_func,
            )

            print_0("Checking correctness for ring attn")

            args_dicts[0]["ring"] = _recursive_to_cuda(args_dicts[0]["ring"])

            ring_out, ring_lse, _ = ring_flash_attn_varlen_kvpacked_func(
                *args_dicts[0]["ring"][0], **args_dicts[0]["ring"][1]
            )

            flash_attn_out_local = extract_local_ring(
                flash_attn_out,
                cu_seqlens,
                dist.get_rank(group=args.cp_group),
                dist.get_world_size(group=args.cp_group),
            )

            out_diff = (flash_attn_out_local - ring_out).abs()
            out_diff_mean = out_diff.mean().item()
            out_diff_max = out_diff.max().item()
            lse_diff_mean, lse_diff_max = get_lse_diff(
                flash_attn_lse, ring_lse, cu_seqlens, "ring", args.cp_group
            )

            print_0("==" * 20)
            _print_diff(
                "Ring Attn",
                "out",
                out_diff_mean,
                out_diff_max,
                print_on_all_devs=True,
            )
            _print_diff(
                "Ring Attn",
                "lse",
                lse_diff_mean,
                lse_diff_max,
                print_on_all_devs=True,
            )
            torch.cuda.empty_cache()
        if args.benchmarks == "te":
            print_0("Checking correctness for te attn")
            # we can only get the output without modifying te code
            seqlens = args_dicts[0]["seqlens"]
            args_dicts[0]["te"] = _recursive_to_cuda(args_dicts[0]["te"])

            te_mod, te_pos_args, te_kwargs = args_dicts[0]["te"]
            te_out = te_mod(*te_pos_args, **te_kwargs)
            te_out = te_out.reshape(te_out.shape[0], -1, args.head_dim)
            flash_attn_out_local = extract_local_zigzag(
                flash_attn_out,
                cu_seqlens,
                dist.get_rank(group=args.cp_group),
                dist.get_world_size(group=args.cp_group),
            )
            flash_attn_out_diff = (flash_attn_out_local - te_out).abs()
            flash_attn_out_diff_mean = flash_attn_out_diff.mean().item()
            flash_attn_out_diff_max = flash_attn_out_diff.max().item()
            print_0("==" * 20)
            _print_diff(
                "TE Attn",
                "out",
                flash_attn_out_diff_mean,
                flash_attn_out_diff_max,
                print_on_all_devs=True,
            )
            torch.cuda.empty_cache()
        if args.benchmarks == "lt":
            print_0("Checking correctness for loongtrain attn")
            seqlens = args_dicts[0]["seqlens"]
            args_dicts[0]["lt"] = _recursive_to_cuda(args_dicts[0]["lt"])

            zero_out_local_padding_fn = args_dicts[0][
                "lt_zero_out_local_padding_fn"
            ]
            get_local_chunk_fn = args_dicts[0]["lt_get_local_chunk_fn"]

            lt_mod, lt_pos_args, lt_kwargs = args_dicts[0]["lt"]
            lt_out = lt_mod(*lt_pos_args, **lt_kwargs)
            world_size = dist.get_world_size(group=args.cp_group)
            lt_out = zero_out_local_padding_fn(lt_out)
            padded_flash_attn_out = unpack_qkv_before_attn(
                flash_attn_out.unsqueeze(0),
                cu_seqlens,
            )
            flash_attn_out_local = get_local_chunk_fn(padded_flash_attn_out)
            lt_out_diff = (flash_attn_out_local - lt_out).abs()
            lt_out_diff_mean = lt_out_diff.mean().item()
            lt_out_diff_max = lt_out_diff.max().item()
            print_0("==" * 20)
            _print_diff(
                "LoongTrain Attn",
                "out",
                lt_out_diff_mean,
                lt_out_diff_max,
                print_on_all_devs=True,
            )
            torch.cuda.empty_cache()
        if args.benchmarks == "dcp":
            print_0("Checking correctness for dcp distributed attn")
            args_dicts[0]["dcp"] = _recursive_to_cuda(args_dicts[0]["dcp"])

            q_shape = args_dicts[0]["q_shape"]
            q_dtype = args_dicts[0]["q_dtype"]
            fw_workload = args_dicts[0]["dcp_fw_workload"]

            raw_seqlens = args_dicts[0]["seqlens"]
            executor, fw_exec_plan, local_q, local_kv = args_dicts[0]["dcp"]
            executor: AttentionExecutor
            executor.fw_exec_plan = fw_exec_plan
            executor.prepare(forward=True)
            executor.init_forward_input(local_q, local_kv)
            out_buffer, lse_buffer = exec_dcp_distributed_attn(
                executor, fw_exec_plan
            )
            cu_raw_seqlens = F.pad(
                torch.cumsum(torch.tensor(raw_seqlens, dtype=torch.int32), 0),
                (1, 0),
            ).to(torch.int32)
            dcp_out, dcp_lse = reconstruct_attention_forward_output_for_test(
                out_buffer,
                lse_buffer,
                q_shape,
                q_dtype,
                cu_raw_seqlens,
                fw_workload,
                fw_exec_plan,
                args.n_devices_per_node,
            )
            out_diff = (flash_attn_out - dcp_out).abs()
            out_diff_mean = out_diff.mean().item()
            out_diff_max = out_diff.max().item()
            lse_diff = (flash_attn_lse - dcp_lse).abs()
            lse_diff_mean = lse_diff.mean().item()
            lse_diff_max = lse_diff.max().item()
            print_0("==" * 20)
            _print_diff(
                "DCP Attn",
                "out",
                out_diff_mean,
                out_diff_max,
                print_on_all_devs=False,
            )
            _print_diff(
                "DCP Attn",
                "lse",
                lse_diff_mean,
                lse_diff_max,
                print_on_all_devs=False,
            )
            torch.cuda.empty_cache()
    else:
        if n_batches == 1:
            args_dicts = list(args_dicts)
            if args.benchmarks == "local":
                print_0("Benchmarking local flash attn")
                local_flash_time = _bench_local_attn(args, args_dicts[0])

                print_0("Local Flash Attn: {:.4f} ms".format(local_flash_time))
                torch.cuda.empty_cache()
            if args.benchmarks == "zigzag":
                print_0("Benchmarking zigzag ring attn")
                zigzag_ring_time = _bench_zigzag(args, args_dicts[0])
                print_0("Zigzag Ring Attn: {:.4f} ms".format(zigzag_ring_time))
                torch.cuda.empty_cache()
            if args.benchmarks == "ring":
                print_0("Benchmarking ring attn")
                ring_time = _bench_ring(args, args_dicts[0])
                print_0("Ring Attn: {:.4f} ms".format(ring_time))
                torch.cuda.empty_cache()
            if args.benchmarks == "te":
                print_0("Benchmarking te attn")
                te_time = _bench_te(args, args_dicts[0])
                print_0("TE Attn: {:.4f} ms".format(te_time))
                torch.cuda.empty_cache()
            if args.benchmarks == "lt":
                print_0("Benchmarking lt attn")
                lt_time = _bench_lt(args, args_dicts[0])
                print_0("LoongTrain Attn: {:.4f} ms".format(lt_time))
                torch.cuda.empty_cache()
            if args.benchmarks == "dcp":
                print_0("Benchmarking dcp distributed attn")
                dcp_time = _bench_dcp(args, args_dicts[0])
                print_0(
                    "DCP Attn: {:.4f} ms".format(
                        dcp_time,
                    )
                )
        else:
            # multi iteration benchmarking
            pbar = (
                tqdm.tqdm(total=n_batches, desc="Benchmark")
                if (dist.get_rank() == 0 and not args.silent)
                else None
            )
            if args.benchmarks == "local":
                print_0("Benchmarking local flash attn")
                local_flash_times = []
                for args_dict in args_dicts:
                    torch.cuda.synchronize()
                    dist.barrier(device_ids=[torch.cuda.current_device()])
                    local_flash_time = _bench_local_attn(args, args_dict)
                    torch.cuda.synchronize()
                    dist.barrier(device_ids=[torch.cuda.current_device()])
                    local_flash_times.append(local_flash_time)
                    if pbar is not None:
                        pbar.update(1)
                local_flash_time_total = sum(local_flash_times)
                print_0(
                    "Local Flash Attn: total time {:.4f} ms".format(
                        local_flash_time_total
                    )
                )
                torch.cuda.empty_cache()
            if args.benchmarks == "zigzag":
                print_0("Benchmarking zigzag ring attn")
                zigzag_ring_times = []
                for args_dict in args_dicts:
                    torch.cuda.synchronize()
                    dist.barrier(device_ids=[torch.cuda.current_device()])
                    zigzag_ring_time = _bench_zigzag(args, args_dict)
                    torch.cuda.synchronize()
                    dist.barrier(device_ids=[torch.cuda.current_device()])
                    zigzag_ring_times.append(zigzag_ring_time)
                    if pbar is not None:
                        pbar.update(1)
                zigzag_ring_time_total = sum(zigzag_ring_times)
                print_0(
                    "Zigzag Ring Attn: total time {:.4f} ms".format(
                        zigzag_ring_time_total
                    )
                )
                torch.cuda.empty_cache()
            if args.benchmarks == "ring":
                print_0("Benchmarking ring attn")
                ring_times = []
                for args_dict in args_dicts:
                    torch.cuda.synchronize()
                    dist.barrier(device_ids=[torch.cuda.current_device()])
                    ring_time = _bench_ring(args, args_dict)
                    torch.cuda.synchronize()
                    dist.barrier(device_ids=[torch.cuda.current_device()])
                    ring_times.append(ring_time)
                    if pbar is not None:
                        pbar.update(1)
                ring_time_total = sum(ring_times)
                print_0(
                    "Ring Attn: total time {:.4f} ms".format(ring_time_total)
                )
                torch.cuda.empty_cache()
            if args.benchmarks == "te":
                print_0("Benchmarking te attn")
                te_times = []
                for args_dict in args_dicts:
                    torch.cuda.synchronize()
                    dist.barrier(device_ids=[torch.cuda.current_device()])
                    te_time = _bench_te(args, args_dict)
                    torch.cuda.synchronize()
                    dist.barrier(device_ids=[torch.cuda.current_device()])
                    te_times.append(te_time)
                    if pbar is not None:
                        pbar.update(1)
                te_time_total = sum(te_times)
                print_0("TE Attn: total time {:.4f} ms".format(te_time_total))
                torch.cuda.empty_cache()
            if args.benchmarks == "lt":
                print_0("Benchmarking lt attn")
                lt_times = []
                for args_dict in args_dicts:
                    torch.cuda.synchronize()
                    dist.barrier(device_ids=[torch.cuda.current_device()])
                    lt_time = _bench_lt(args, args_dict)
                    torch.cuda.synchronize()
                    dist.barrier(device_ids=[torch.cuda.current_device()])
                    lt_times.append(lt_time)
                    if pbar is not None:
                        pbar.update(1)
                lt_time_total = sum(lt_times)
                print_0(
                    "LoongTrain Attn: total time {:.4f} ms".format(
                        lt_time_total
                    )
                )
                torch.cuda.empty_cache()
            if args.benchmarks == "dcp":
                print_0("Benchmarking dcp distributed attn")
                dcp_times = []
                for args_dict in args_dicts:
                    torch.cuda.synchronize()
                    dist.barrier(device_ids=[torch.cuda.current_device()])
                    dcp_time = _bench_dcp(args, args_dict)
                    torch.cuda.synchronize()
                    dist.barrier(device_ids=[torch.cuda.current_device()])
                    dcp_times.append(dcp_time)
                    if pbar is not None:
                        pbar.update(1)
                dcp_time_total = sum(dcp_times)
                print_0(
                    "DCP Attn: total time {:.4f} ms".format(dcp_time_total)
                )
                torch.cuda.empty_cache()
    dist.destroy_process_group()
