import pickle
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.benchmark as benchmark

from dcp.ops import _flash_attn_update_out_lse


@torch.jit.script
def _update_out_and_lse(
    out: torch.Tensor,
    lse: torch.Tensor,
    block_out: torch.Tensor,
    block_lse: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:

    block_out = block_out.to(torch.float32)
    block_lse = block_lse.transpose(-2, -1).unsqueeze(dim=-1)
    lse = lse.transpose(-2, -1).unsqueeze(dim=-1)

    # new_lse = lse + torch.log(1 + torch.exp(block_lse - lse))
    # torch.exp(lse - new_lse) * out + torch.exp(block_lse - new_lse) * block_out
    # For additional context and discussion, please refer to:
    # https://github.com/zhuzilin/ring-flash-attention/pull/34#issuecomment-2076126795
    out = out - F.sigmoid(block_lse - lse) * (out - block_out)
    lse = lse - F.logsigmoid(lse - block_lse)

    return out, lse.squeeze().transpose(-2, -1)


def ref_update_out_lse(
    src_out_buffer: torch.Tensor,
    dst_out_buffer: torch.Tensor,
    src_lse_buffer: torch.Tensor,
    dst_lse_buffer: torch.Tensor,
    src_block_table: torch.Tensor,
    dst_block_table: torch.Tensor,
    block_lengths: torch.Tensor,
):
    n_ops = len(block_lengths)
    for op_id in range(n_ops):
        src_block_indices = src_block_table[
            op_id, : block_lengths[op_id].item()
        ].tolist()
        dst_block_index = dst_block_table[op_id].item()
        for block_idx, src_indice in enumerate(src_block_indices):
            src_out_block = src_out_buffer[
                src_indice
            ]  # [block_size, n_head, head_dim]
            dst_out_block = dst_out_buffer[
                dst_block_index
            ]  # [block_size, n_head, head_dim]

            src_lse_block = src_lse_buffer[
                :, src_indice
            ]  # [n_head, block_size]
            dst_lse_block = dst_lse_buffer[
                :, dst_block_index
            ]  # [n_head, block_size]
            if block_idx == 0:
                (
                    dst_out_buffer[dst_block_index],
                    dst_lse_buffer[:, dst_block_index],
                ) = (src_out_block, src_lse_block)
            else:
                (
                    dst_out_buffer[dst_block_index],
                    dst_lse_buffer[:, dst_block_index],
                ) = _update_out_and_lse(
                    dst_out_block, dst_lse_block, src_out_block, src_lse_block
                )


def benchmark_forward(
    fn,
    *inputs,
    repeats=10,
    desc="",
    verbose=True,
    amp=False,
    amp_dtype=torch.float16,
    **kwinputs,
):
    """Use Pytorch Benchmark on the forward pass of an arbitrary function."""
    if verbose:
        print(desc, "- Forward pass")

    def amp_wrapper(*inputs, **kwinputs):
        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp):
            fn(*inputs, **kwinputs)

    t = benchmark.Timer(
        stmt="fn_amp(*inputs, **kwinputs)",
        globals={
            "fn_amp": amp_wrapper,
            "inputs": inputs,
            "kwinputs": kwinputs,
        },
        num_threads=torch.get_num_threads(),
    )
    m = t.timeit(repeats)
    if verbose:
        print(m)
    return t, m


def benchmerk_flash_attn_update_out_lse(
    n_ops: int,
    n_src_blocks_per_op: int,
    n_head: int,
    head_dim: int,
    triton_block_size: int,
):
    # returns us
    buffer_size_n_blocks = 2 * (n_ops * n_src_blocks_per_op)
    torch.set_default_device("cuda")
    with torch.no_grad():
        block_size = 256
        src_out_buffer = torch.randn(
            buffer_size_n_blocks, block_size, n_head, head_dim
        )
        dst_out_buffer = torch.zeros(
            buffer_size_n_blocks, block_size, n_head, head_dim
        )
        src_lse_buffer = torch.randn(n_head, buffer_size_n_blocks, block_size)
        dst_lse_buffer = torch.zeros(n_head, buffer_size_n_blocks, block_size)

        src_block_table = torch.tensor(
            np.random.choice(
                buffer_size_n_blocks,
                ((n_ops, n_src_blocks_per_op)),
                replace=False,
            ),
            dtype=torch.int32,
        )
        dst_block_table = torch.tensor(
            np.random.choice(
                buffer_size_n_blocks,
                ((n_ops,)),
                replace=False,
            ),
            dtype=torch.int32,
        )

        def _get_grid_dims():
            assert block_size % triton_block_size == 0
            n_programs_per_block = block_size // triton_block_size
            return (n_ops * n_head * n_programs_per_block,)

        block_lengths_tensor = torch.tensor(
            [n_src_blocks_per_op] * n_ops, dtype=torch.int32
        )

        ref_dst_out_buffer = dst_out_buffer.clone()
        ref_dst_lse_buffer = dst_lse_buffer.clone()

        grid_dims = _get_grid_dims()

        def _run_triton():
            _flash_attn_update_out_lse[grid_dims](
                src_out_buffer,
                dst_out_buffer,
                src_lse_buffer,
                dst_lse_buffer,
                n_head,
                src_block_table,
                dst_block_table,
                block_lengths_tensor,
                block_size,
                block_size * n_head * head_dim,
                n_head * head_dim,
                block_size,
                1,
                buffer_size_n_blocks * block_size,
                buffer_size_n_blocks * block_size,
                head_dim,
                n_src_blocks_per_op,
                triton_block_size,
            )

        def _run_pytorch():
            ref_update_out_lse(
                src_out_buffer,
                ref_dst_out_buffer,
                src_lse_buffer,
                ref_dst_lse_buffer,
                src_block_table,
                dst_block_table,
                block_lengths_tensor,
            )

        triton_f = (
            benchmark_forward(
                _run_triton,
                repeats=30,
                verbose=False,
            )[1].mean
            * 1e6
        )

        pytorch_f = (
            benchmark_forward(
                _run_pytorch,
                repeats=30,
                verbose=False,
            )[1].mean
            * 1e6
        )
    return triton_f, pytorch_f


results = {}

for n_ops in [1, 2, 4, 8, 16, 32, 64]:
    for n_src_blocks_per_op in [2, 4, 8, 16, 32]:
        n_head = 8
        head_dim = 128
        for triton_block_size in [32, 64]:
            config = (
                n_ops,
                n_src_blocks_per_op,
                n_head,
                head_dim,
                triton_block_size,
            )
            try:
                triton_f, pytorch_f = benchmerk_flash_attn_update_out_lse(
                    n_ops,
                    n_src_blocks_per_op,
                    n_head,
                    head_dim,
                    triton_block_size,
                )
                results[config, "Triton"] = triton_f
                results[config, "Pytorch"] = pytorch_f
                print(
                    "============ n_ops: {n_ops}, n_src_blocks_per_op: {n_src_blocks_per_op}, triton_block_size: {triton_block_size}"
                )
                print(
                    "Triton time:",
                    triton_f,
                    "us,",
                    "PyTorch time:",
                    pytorch_f,
                    "us",
                )
            except Exception as e:
                print(f"Failed to run config: {config}, error: {e}")

with open("benchmark_fused_update.pkl", "wb") as f:
    pickle.dump(results, f, pickle.HIGHEST_PROTOCOL)
