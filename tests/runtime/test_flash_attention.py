import argparse
import os
import math

import numpy as np
import termcolor
import torch
import torch.distributed as dist
import torch.nn.functional as F
from einops import rearrange, repeat
from flash_attn import (
    flash_attn_varlen_kvpacked_func as flash_attn_varlen_kvpacked_func_ref,
)
from dcp_flash_attn import (
    flash_attn_varlen_kvpacked_func as dcp_flash_attn_varlen_kvpacked_func,
)
from ring_flash_attn import zigzag_ring_flash_attn_varlen_kvpacked_func
import transformer_engine as te

from dcp.runtime.flash_attention import (
    prepare_dcp_distributed_attn_for_test,
)
from dcp.runtime.flash_attention.utils import (
    reconstruct_attention_backward_output_for_test,
    reconstruct_attention_forward_output_for_test,
    unpad_attention_output,
    unpad_attention_grad,
    pad_q_kv,
    pad_q_kv_dout_to_max_seqlen,
    pad_dout,
    unpad_input,
    causal_range_mask_fn,
    lambda_mask_fn,
    modality_specific_mask_fn,
    get_local_attn_ranges_zigzag,
    get_per_step_attn_range,
)


_MASK_FN_DICT = {
    "causal": None,
    "lambda": lambda_mask_fn,
    "multimodal": modality_specific_mask_fn,
}


def get_mask_fn(mask_type):
    if mask_type not in _MASK_FN_DICT:
        raise ValueError(f"Unsupported mask type {mask_type}")
    return _MASK_FN_DICT[mask_type]


def print_all(s):
    for rank in range(dist.get_world_size()):
        if rank == dist.get_rank():
            print(f"RANK {rank}: {s}")
        dist.barrier(device_ids=[torch.cuda.current_device()])


def print_0(s):
    if dist.get_rank() == 0:
        print(s)
    dist.barrier(device_ids=[torch.cuda.current_device()])


def extract_lse(lse, cu_seqlens):
    values = []
    for i in range(len(cu_seqlens) - 1):
        start, end = cu_seqlens[i], cu_seqlens[i + 1]
        value = lse[i, :, : end - start]
        values.append(value)
    return values


def init_env():
    local_rank = os.environ.get("LOCAL_RANK")
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    torch.set_default_device(device)
    print_all(f"Rank {dist.get_rank()} set device {device}")


def construct_local_mask(
    seqlen_q,
    seqlen_k,
    window_size=(-1, -1),  # -1 means infinite window size
    query_padding_mask=None,
    key_padding_mask=None,
    device=None,
    key_leftpad=None,
):
    row_idx = rearrange(
        torch.arange(seqlen_q, device=device, dtype=torch.long), "s -> s 1"
    )
    col_idx = torch.arange(seqlen_k, device=device, dtype=torch.long)
    if key_leftpad is not None:
        key_leftpad = rearrange(key_leftpad, "b -> b 1 1 1")
        col_idx = repeat(col_idx, "s -> b 1 1 s", b=key_leftpad.shape[0])
        col_idx = torch.where(
            col_idx >= key_leftpad, col_idx - key_leftpad, 2**32
        )
    sk = (
        seqlen_k
        if key_padding_mask is None
        else rearrange(key_padding_mask.sum(-1), "b -> b 1 1 1")
    )
    sq = (
        seqlen_q
        if query_padding_mask is None
        else rearrange(query_padding_mask.sum(-1), "b -> b 1 1 1")
    )
    if window_size[0] < 0:
        return col_idx > row_idx + sk - sq + window_size[1]
    else:
        sk = (
            torch.full_like(col_idx, seqlen_k)
            if key_padding_mask is None
            else sk
        )
        return torch.logical_or(
            col_idx > torch.minimum(row_idx + sk - sq + window_size[1], sk),
            col_idx < row_idx + sk - sq - window_size[0],
        )


# reference pytorch attn impl from flash attn
def attention_ref(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    query_padding_mask=None,
    key_padding_mask=None,
    attn_bias=None,
    dropout_p=0.0,
    dropout_mask=None,
    causal=False,
    window_size=(-1, -1),  # -1 means infinite window size
    softcap=0.0,
    upcast=True,
    upcast_no_downcast=False,
    reorder_ops=False,
    key_leftpad=None,
    attn_range=None,
):
    """
    Arguments:
        q: (batch_size, seqlen_q, nheads, head_dim)
        k: (batch_size, seqlen_k, nheads_k, head_dim)
        v: (batch_size, seqlen_k, nheads_k, head_dim)
        query_padding_mask: (batch_size, seqlen_q)
        key_padding_mask: (batch_size, seqlen_k)
        attn_bias: broadcastable to (batch_size, nheads, seqlen_q, seqlen_k)
        dropout_p: float
        dropout_mask: (batch_size, nheads, seqlen_q, seqlen_k)
        causal: whether to apply causal masking
        window_size: (int, int), left and right window size
        upcast: whether to cast all inputs to fp32, do all computation in fp32, then cast
            output back to fp16/bf16.
        reorder_ops: whether to change the order of operations (scaling k instead of scaling q, etc.)
            without changing the math. This is to estimate the numerical error from operation
            reordering.
    Output:
        output: (batch_size, seqlen_q, nheads, head_dim)
        attention: (batch_size, nheads, seqlen_q, seqlen_k), softmax after dropout
    """
    if causal:
        window_size = (window_size[0], 0)
    dtype_og = q.dtype
    b = q.shape[0]
    seqlen_q = q.shape[1]
    seqlen_k = k.shape[1]
    if upcast:
        q, k, v = q.float(), k.float(), v.float()
    if attn_range is not None:
        mask = torch.zeros(
            b, seqlen_q, seqlen_k, device=q.device, dtype=torch.bool
        )
        if attn_range.dim() == 3:
            assert attn_range.shape[0] == 2
            attn_ranges = [attn_range[0], attn_range[1]]
        else:
            attn_ranges = [attn_range]
        for atn_range in attn_ranges:
            curr_mask = torch.zeros(
                b, seqlen_q, seqlen_k, device=q.device, dtype=torch.bool
            )
            # atn_range is of shape (2, b * seqlen_q)
            b = q.shape[0]
            seqlen_q = q.shape[1]
            cum_seqlens = 0
            for i in range(b):
                curr_seqlen = query_padding_mask[i].sum().item()
                for q_id in range(curr_seqlen):
                    min_id, max_id = (
                        atn_range[0, cum_seqlens + q_id],
                        atn_range[1, cum_seqlens + q_id],
                    )
                    curr_mask[i, q_id, min_id:max_id] = True
                cum_seqlens += curr_seqlen
            mask = mask | curr_mask
        range_mask = rearrange(mask, "b s l -> b 1 s l")
    else:
        range_mask = None
    seqlen_q, seqlen_k = q.shape[1], k.shape[1]
    k = repeat(k, "b s h d -> b s (h g) d", g=q.shape[2] // k.shape[2])
    v = repeat(v, "b s h d -> b s (h g) d", g=q.shape[2] // v.shape[2])
    d = q.shape[-1]
    if not reorder_ops:
        scores = torch.einsum("bthd,bshd->bhts", q / math.sqrt(d), k)
    else:
        scores = torch.einsum("bthd,bshd->bhts", q, k / math.sqrt(d))
    if softcap > 0:
        scores = scores / softcap
        scores = scores.tanh()
        scores = scores * softcap
    attn_mask = None
    if key_padding_mask is not None:
        attn_mask = rearrange(~key_padding_mask, "b s -> b 1 1 s")
        # scores.masked_fill_(rearrange(~key_padding_mask, "b s -> b 1 1 s"), float("-inf"))
    if query_padding_mask is not None:
        if attn_mask is None:
            attn_mask = rearrange(~query_padding_mask, "b s -> b 1 s 1")
        else:
            attn_mask = attn_mask | rearrange(
                ~query_padding_mask, "b s -> b 1 s 1"
            )
    if window_size[0] >= 0 or window_size[1] >= 0:
        local_mask = construct_local_mask(
            seqlen_q,
            seqlen_k,
            window_size,
            query_padding_mask,
            key_padding_mask,
            q.device,
            key_leftpad=key_leftpad,
        )
        if attn_mask is None:
            attn_mask = local_mask
        else:
            attn_mask = attn_mask | local_mask
    if range_mask is not None:
        if attn_mask is None:
            attn_mask = range_mask
        else:
            attn_mask = (~range_mask) | attn_mask
    if attn_mask is not None:
        scores.masked_fill_(attn_mask, float("-inf"))

    if attn_bias is not None:
        scores = scores + attn_bias
    attention = torch.softmax(scores, dim=-1).to(v.dtype)
    # Some rows might be completely masked out so we fill them with zero instead of NaN
    if attn_mask is not None:
        attention = attention.masked_fill(
            torch.all(attn_mask, dim=-1, keepdim=True), 0.0
        )
    # We want to mask here so that the attention matrix doesn't have any NaNs
    # Otherwise we'll get NaN in dV
    if query_padding_mask is not None:
        attention = attention.masked_fill(
            rearrange(~query_padding_mask, "b s -> b 1 s 1"), 0.0
        )
    dropout_scaling = 1.0 / (1 - dropout_p)
    # attention_drop = attention.masked_fill(~dropout_mask, 0.0) * dropout_scaling
    # output = torch.einsum('bhts,bshd->bthd', attention_drop , v)
    if dropout_mask is not None:
        attention_drop = attention.masked_fill(~dropout_mask, 0.0)
    else:
        attention_drop = attention
    output = torch.einsum(
        "bhts,bshd->bthd", attention_drop, v * dropout_scaling
    )
    if query_padding_mask is not None:
        output.masked_fill_(
            rearrange(~query_padding_mask, "b s -> b s 1 1"), 0.0
        )
    if upcast_no_downcast:
        return output, attention
    return output.to(dtype=dtype_og), attention.to(dtype=dtype_og)


def exec_dcp_distributed_attn(
    q,
    kv,
    attn_mask_type,
    dout,
    raw_seqlens,
    args,
    text_lengths=None,
    image_sizes=None,
):
    q = q.detach()
    kv = kv.detach()
    (
        data_loader,  # data_loader should not destruct before test ends
        executor,
        fw_exec_plan,
        bw_exec_plan,
        local_q,
        local_kv,
        local_dout,
        fw_workload,
        bw_workload,
    ) = prepare_dcp_distributed_attn_for_test(
        q,
        kv,
        attn_mask_type,
        raw_seqlens,
        args.n_devices_per_node,
        args.block_size,
        args.head_block_size,
        dout=dout if not args.skip_backward else None,
        mem_imbalance_epsilon=args.mem_epsilon,
        comp_imbalance_epsilon=args.comp_epsilon,
        inter_node_comp_imbalance_factor=args.inter_node_comp_imbalance_factor,
        dropout_p=args.dropout_p,
        deterministic=args.deterministic,
        synchronous=args.synchronous,
        mask_fn=get_mask_fn(attn_mask_type),
        text_lengths=text_lengths,
        img_sizes=image_sizes,
    )
    executor.init_buffers(fw_exec_plan)
    executor.init_forward_input(local_q, local_kv)
    executor.fw_exec_plan = fw_exec_plan
    executor.prepare(forward=True)
    out_buffer, lse_buffer = executor.execute(fw_exec_plan)
    torch.cuda.synchronize()
    cumsum_raw_seqlens = torch.tensor(
        np.cumsum([0] + raw_seqlens), dtype=torch.int32
    )
    global_out, global_lse = reconstruct_attention_forward_output_for_test(
        out_buffer,
        lse_buffer,
        q.shape,
        q.dtype,
        cumsum_raw_seqlens,
        fw_workload,
        fw_exec_plan,
        args.n_devices_per_node,
    )
    torch.cuda.synchronize()
    if not args.skip_backward:
        executor.deallocate_buffers()
        # now copy dout to the output buffer
        executor.init_buffers(bw_exec_plan)
        executor.init_backward_input(local_dout)
        executor.bw_exec_plan = bw_exec_plan
        executor.prepare(forward=False)
        dq_buffer, dkv_buffer = executor.execute(
            bw_exec_plan, is_forward=False
        )
        global_dq, global_dkv = reconstruct_attention_backward_output_for_test(
            dq_buffer,
            dkv_buffer,
            q,
            kv,
            bw_workload,
            bw_exec_plan,
            cumsum_raw_seqlens,
            args.n_devices_per_node,
        )
    else:
        global_dq = None
        global_dkv = None

    return global_out, global_lse, global_dq, global_dkv, executor


def exec_local_flash_attn(
    q, kv, dout, raw_seqlens, args, text_lengths=None, image_sizes=None
):
    assert args.attn_mask_type == "causal"
    assert text_lengths is None
    assert image_sizes is None
    d = q.shape[-1]
    softmax_scale = d ** (-0.5)
    cu_seqlens = F.pad(
        torch.cumsum(torch.tensor(raw_seqlens, dtype=torch.int32), 0), (1, 0)
    ).to(torch.int32)
    out, lse, _ = flash_attn_varlen_kvpacked_func_ref(
        q,
        kv,
        cu_seqlens,
        cu_seqlens,
        max(raw_seqlens),
        max(raw_seqlens),
        args.dropout_p,
        softmax_scale,
        causal=True,  # TODO: fix this test later
        deterministic=args.deterministic,
        return_attn_probs=True,
    )
    (
        dq,
        dkv,
    ) = torch.autograd.grad(out, (q, kv), dout)
    return out, lse, dq, dkv


def exec_local_pytorch_attn(
    q, kv, dout, raw_seqlens, args, text_lengths=None, image_sizes=None
):
    assert args.attn_mask_type != "causal"
    if args.attn_mask_type == "lambda":
        attn_mask_fn = lambda_mask_fn
        mask_fn_info = None
    elif args.attn_mask_type == "multimodal":
        assert text_lengths is not None
        assert image_sizes is not None
        attn_mask_fn = modality_specific_mask_fn
        mask_fn_info = []
        for i in range(len(raw_seqlens)):
            mask_fn_info.append(
                {
                    "text_lengths": text_lengths[i],
                    "image_size": image_sizes[i],
                }
            )
    attn_mask = attn_mask_fn(
        raw_seqlens, raw_seqlens, torch.int32, mask_fn_info
    )
    if attn_mask.dim() == 3:
        attn_mask = attn_mask.permute(1, 2, 0).contiguous()
    else:
        attn_mask = attn_mask.transpose(1, 0).contiguous()
    # pad q kv to [batch_size, max_seqlen, n_heads, head_dim]
    max_seqlen = max(raw_seqlens)
    padded_q, padded_kv, padded_dout, padding_mask = (
        pad_q_kv_dout_to_max_seqlen(q, kv, dout, raw_seqlens, max_seqlen)
    )
    padded_k = padded_kv[:, :, 0].squeeze(2)
    padded_v = padded_kv[:, :, 1].squeeze(2)
    out, _ = attention_ref(
        padded_q,
        padded_k,
        padded_v,
        query_padding_mask=padding_mask,
        key_padding_mask=padding_mask,
        attn_range=attn_mask,
    )
    dq, dk, dv = torch.autograd.grad(
        out, (padded_q, padded_k, padded_v), padded_dout
    )
    dkv = torch.stack([dk, dv], dim=2)
    # unpad input/outputs
    unpad_out = unpad_input(out, padding_mask)
    unpad_dq = unpad_input(dq, padding_mask)
    unpad_dkv = unpad_input(dkv, padding_mask)
    return unpad_out, None, unpad_dq, unpad_dkv


def exec_local_dcp_flash_attn(
    q, kv, dout, raw_seqlens, args, text_lengths=None, image_sizes=None
):
    assert args.attn_mask_type != "causal"
    if args.attn_mask_type == "lambda":
        attn_mask_fn = lambda_mask_fn
        mask_fn_info = None
    elif args.attn_mask_type == "multimodal":
        attn_mask_fn = modality_specific_mask_fn
        mask_fn_info = []
        for i in range(len(raw_seqlens)):
            mask_fn_info.append(
                {
                    "text_lengths": text_lengths[i],
                    "image_size": image_sizes[i],
                }
            )
    attn_mask = attn_mask_fn(
        raw_seqlens, raw_seqlens, torch.int32, mask_fn_info
    )
    if attn_mask.dim() == 3:
        attn_mask = attn_mask.permute(1, 2, 0).contiguous()
    else:
        attn_mask = attn_mask.transpose(1, 0).contiguous()
    d = q.shape[-1]
    softmax_scale = d ** (-0.5)
    cu_seqlens = F.pad(
        torch.cumsum(torch.tensor(raw_seqlens, dtype=torch.int32), 0), (1, 0)
    ).to(torch.int32)
    out, lse, _ = dcp_flash_attn_varlen_kvpacked_func(
        q,
        kv,
        cu_seqlens,
        cu_seqlens,
        sum(raw_seqlens),
        sum(raw_seqlens),
        max(raw_seqlens),
        max(raw_seqlens),
        args.dropout_p,
        softmax_scale,
        return_attn_probs=True,
        attn_range=attn_mask,
        force_split_kv=True,
    )
    (
        dq,
        dkv,
    ) = torch.autograd.grad(out, (q, kv), dout)
    return out, lse, dq, dkv


def exec_te_attn(
    q, kv, dout, raw_seqlens, args, text_lengths=None, image_sizes=None
):
    te_attn_module = te.pytorch.DotProductAttention(
        num_attention_heads=q.shape[-2],
        kv_channels=q.shape[-1],  # same as head_dim,
        num_gqa_groups=kv.shape[-2],
        attn_mask_type="custom_ranges",
        qkv_format="thd",
        cp_group=torch.distributed.group.WORLD,
        cp_global_ranks=list(range(dist.get_world_size())),
        cp_stream=torch.cuda.Stream(),
        cp_comm_type="p2p",
    )
    d = q.shape[-1]
    softmax_scale = d ** (-0.5)
    world_size = dist.get_world_size()
    padded_seqlens = [
        (seqlen + (2 * world_size) - 1) // (2 * world_size) * (2 * world_size)
        for seqlen in raw_seqlens
    ]
    padded_cu_seqlens = F.pad(
        torch.cumsum(torch.tensor(padded_seqlens, dtype=torch.int32), 0),
        (1, 0),
    ).to(torch.int32)
    padded_q, padded_kv = pad_q_kv(q, kv, raw_seqlens, padded_seqlens)
    padded_dout = pad_dout(dout, raw_seqlens, padded_seqlens)

    assert torch.all(padded_cu_seqlens % (2 * world_size) == 0)
    max_seqlen = max(padded_seqlens)
    local_q = extract_local_zigzag(
        padded_q, padded_cu_seqlens, dist.get_rank(), world_size
    )
    local_kv = extract_local_zigzag(
        padded_kv, padded_cu_seqlens, dist.get_rank(), world_size
    )
    local_q.requires_grad_()
    local_kv.requires_grad_()
    local_cu_seqlens = padded_cu_seqlens // world_size
    local_max_seqlen = max_seqlen // world_size
    q = local_q
    k = local_kv[:, 0].squeeze()
    v = local_kv[:, 1].squeeze()

    if args.attn_mask_type == "causal":
        attn_mask_fn = causal_range_mask_fn
        mask_fn_info = None
    elif args.attn_mask_type == "lambda":
        attn_mask_fn = lambda_mask_fn
        mask_fn_info = None
    elif args.attn_mask_type == "multimodal":
        attn_mask_fn = modality_specific_mask_fn
        mask_fn_info = []
        for i in range(len(raw_seqlens)):
            mask_fn_info.append(
                {
                    "text_lengths": text_lengths[i],
                    "image_size": image_sizes[i],
                }
            )
    else:
        raise ValueError(f"Unsupported mask type {args.attn_mask_type}")
    global_attn_ranges = attn_mask_fn(
        raw_seqlens, padded_seqlens, torch.int32, mask_fn_info
    )
    if global_attn_ranges.dim() == 3:
        global_attn_ranges = global_attn_ranges.permute(1, 2, 0).contiguous()
    else:
        global_attn_ranges = global_attn_ranges.transpose(1, 0).contiguous()
    local_attn_ranges = get_local_attn_ranges_zigzag(
        global_attn_ranges, padded_seqlens, world_size, dist.get_rank()
    ).cpu()
    padded_cu_seqlens_cpu = padded_cu_seqlens.cpu()
    per_step_attn_ranges = [
        get_per_step_attn_range(
            i,
            dist.get_rank(),
            world_size,
            local_attn_ranges,
            padded_cu_seqlens_cpu,
        )
        for i in range(world_size)
    ]
    per_step_attn_ranges = [ranges.cuda() for ranges in per_step_attn_ranges]
    output = te_attn_module(
        q,
        k,
        v,
        core_attention_bias_type="no_bias",
        cu_seqlens_q=padded_cu_seqlens,
        cu_seqlens_kv=padded_cu_seqlens,
        max_seqlen_q=max_seqlen,
        max_seqlen_kv=max_seqlen,
        attn_ranges_per_step=per_step_attn_ranges,
    )
    if not args.skip_backward:
        local_dout = extract_local_zigzag(
            padded_dout, padded_cu_seqlens, dist.get_rank(), world_size
        )
        local_dout = local_dout.reshape(output.shape)
        (dq, dk, dv) = torch.autograd.grad(output, (q, k, v), local_dout)
        dkv = torch.stack([dk, dv], dim=1)
    else:
        dq, dkv = None, None
    return (
        output.reshape(-1, q.shape[-2], q.shape[-1]),
        None,
        dq,
        dkv,
        padded_seqlens,
    )


def extract_local_zigzag(value, cu_seqlens, rank, world_size):
    local_values = []
    for i in range(len(cu_seqlens) - 1):
        start, end = cu_seqlens[i].item(), cu_seqlens[i + 1].item()
        local_value = value[start:end].chunk(2 * world_size, dim=0)
        local_values.extend(
            [
                local_value[rank].detach().clone(),
                local_value[2 * world_size - 1 - rank].detach().clone(),
            ]
        )
    return torch.cat(local_values, dim=0).contiguous()


def _put_back_local_zigzag_for_rank(
    local_value, output_value, global_cu_seqlens, rank, world_size
):
    local_cu_seqlens = global_cu_seqlens // (2 * world_size)
    for i in range(len(global_cu_seqlens) - 1):
        global_start = global_cu_seqlens[i].item()
        local_start = local_cu_seqlens[i].item() * 2
        chunk_size = (local_cu_seqlens[i + 1] - local_cu_seqlens[i]).item()
        global_out_offset = global_start + chunk_size * rank
        local_offset = local_start
        output_value[global_out_offset : global_out_offset + chunk_size] = (
            local_value[local_offset : local_offset + chunk_size]
        )
        global_out_offset1 = global_start + chunk_size * (
            2 * world_size - 1 - rank
        )
        local_offset1 = local_start + chunk_size
        output_value[global_out_offset1 : global_out_offset1 + chunk_size] = (
            local_value[local_offset1 : local_offset1 + chunk_size]
        )
    return output_value


def put_back_local_zigzag(
    global_values, local_buffer, global_cu_seqlens, world_size
):
    for rank in range(world_size):
        local_buffer = _put_back_local_zigzag_for_rank(
            global_values[rank],
            local_buffer,
            global_cu_seqlens,
            rank,
            world_size,
        )
    return global_values


def exec_ring_flash_attn(q, kv, dout, raw_seqlens, args):
    d = q.shape[-1]
    softmax_scale = d ** (-0.5)
    world_size = dist.get_world_size()
    padded_seqlens = [
        (seqlen + (2 * world_size) - 1) // (2 * world_size) * (2 * world_size)
        for seqlen in raw_seqlens
    ]
    padded_cu_seqlens = F.pad(
        torch.cumsum(torch.tensor(padded_seqlens, dtype=torch.int32), 0),
        (1, 0),
    ).to(torch.int32)
    padded_q, padded_kv = pad_q_kv(q, kv, raw_seqlens, padded_seqlens)
    padded_dout = pad_dout(dout, raw_seqlens, padded_seqlens)

    assert torch.all(padded_cu_seqlens % (2 * world_size) == 0)
    max_seqlen = max(padded_seqlens)
    local_q = extract_local_zigzag(
        padded_q, padded_cu_seqlens, dist.get_rank(), world_size
    )
    local_kv = extract_local_zigzag(
        padded_kv, padded_cu_seqlens, dist.get_rank(), world_size
    )
    local_q.requires_grad_()
    local_kv.requires_grad_()
    local_cu_seqlens = padded_cu_seqlens // world_size
    local_max_seqlen = max_seqlen // world_size
    # for now we assume attn_mask is causal mask
    out, lse, _ = zigzag_ring_flash_attn_varlen_kvpacked_func(
        local_q,
        local_kv,
        local_cu_seqlens,
        local_max_seqlen,
        args.dropout_p,
        softmax_scale,
        causal=True,  # TODO: fix this test later
        deterministic=args.deterministic,
        return_attn_probs=True,
    )
    local_dout = extract_local_zigzag(
        padded_dout, padded_cu_seqlens, dist.get_rank(), world_size
    )
    (dq, dkv) = torch.autograd.grad(out, (local_q, local_kv), local_dout)
    return out, lse, dq, dkv, padded_seqlens


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-nd", "--n-devices-per-node", type=int, default=8)
    parser.add_argument("-me", "--mem-epsilon", type=float, default=0.2)
    parser.add_argument("-ce", "--comp-epsilon", type=float, default=0.2)
    parser.add_argument(
        "--inter-node-comp-imbalance-factor", type=float, default=5.0
    )
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--block-size", type=int, default=512)
    parser.add_argument("--head-block-size", type=int, default=1)
    parser.add_argument("--dropout-p", type=float, default=0.0)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--n-query-groups", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--synchronous", action="store_true")
    parser.add_argument("--skip-backward", action="store_true")
    parser.add_argument(
        "-m",
        "--attn-mask-type",
        type=str,
        default="causal",
        choices=["causal", "lambda", "multimodal"],
    )
    args = parser.parse_args()

    assert (
        args.n_heads % args.n_query_groups == 0
    ), "n_heads should be divisible by n_query_groups"

    if args.dtype == "bfloat16":
        args.dtype = torch.bfloat16
    elif args.dtype == "float16":
        args.dtype = torch.float16
    else:
        raise ValueError("Unsupported dtype {}".format(args.dtype))

    dist.init_process_group(backend="nccl")
    init_env()
    if args.n_devices_per_node > dist.get_world_size():
        args.n_devices_per_node = dist.get_world_size()
        if dist.get_rank() == 0:
            print(f"Setting n_devices_per_node to {args.n_devices_per_node}.")
    raw_seqlens = [1050, 1643, 1050, 1643]  # , 2434, 2443, 3543, 2311]
    text_lengths = (
        [
            [512, 25],
            [483, 79, 57],
            [512, 25],
            [483, 79, 57],
        ]
        if args.attn_mask_type == "multimodal"
        else None
    )
    image_sizes = (
        [512, 512, 512, 512] if args.attn_mask_type == "multimodal" else None
    )
    # raw_seqlens = [65536]
    torch.manual_seed(args.seed)
    q = torch.randn(
        sum(raw_seqlens), args.n_heads, args.head_dim, dtype=args.dtype
    ).requires_grad_()
    kv = torch.randn(
        sum(raw_seqlens),
        2,
        args.n_query_groups,
        args.head_dim,
        dtype=args.dtype,
    ).requires_grad_()
    dout = torch.randn(
        sum(raw_seqlens), args.n_heads, args.head_dim, dtype=args.dtype
    )
    cu_seqlens = F.pad(
        torch.cumsum(torch.tensor(raw_seqlens, dtype=torch.int32), 0), (1, 0)
    ).to(torch.int32)
    with torch.no_grad():
        dist.broadcast(q, src=0)
        dist.broadcast(kv, src=0)
        dist.broadcast(dout, src=0)
    if args.attn_mask_type == "causal":
        ref_out, ref_lse, ref_dq, ref_dkv = exec_local_flash_attn(
            q, kv, dout, raw_seqlens, args, text_lengths, image_sizes
        )
    else:
        ref_out, ref_lse, ref_dq, ref_dkv = exec_local_dcp_flash_attn(
            q, kv, dout, raw_seqlens, args, text_lengths, image_sizes
        )
    out, lse, dq, dkv, exec = exec_dcp_distributed_attn(
        q,
        kv,
        args.attn_mask_type,
        dout,
        raw_seqlens,
        args,
        text_lengths,
        image_sizes,
    )
    (
        ring_local_out,
        ring_local_lse,
        ring_local_dq,
        ring_local_dkv,
        ring_padded_seqlens,
    ) = exec_te_attn(q, kv, dout, raw_seqlens, args, text_lengths, image_sizes)
    ref_local_out = extract_local_zigzag(
        ref_out, cu_seqlens, dist.get_rank(), dist.get_world_size()
    )

    # reconstruct the global output
    ring_cu_padded_seqlens = F.pad(
        torch.cumsum(torch.tensor(ring_padded_seqlens, dtype=torch.int32), 0),
        (1, 0),
    ).to(torch.int32)
    ring_global_outs = [
        torch.zeros_like(ring_local_out) for _ in range(dist.get_world_size())
    ]
    dist.all_gather(ring_global_outs, ring_local_out)

    ring_global_out = torch.zeros(
        sum(ring_padded_seqlens),
        args.n_heads,
        args.head_dim,
        dtype=args.dtype,
    )
    put_back_local_zigzag(
        ring_global_outs,
        ring_global_out,
        ring_cu_padded_seqlens,
        dist.get_world_size(),
    )

    if ring_local_lse is not None:
        ring_global_lses = [
            torch.zeros_like(ring_local_lse)
            for _ in range(dist.get_world_size())
        ]
        dist.all_gather(ring_global_lses, ring_local_lse)
        ring_global_lse = torch.zeros(
            sum(ring_padded_seqlens), args.n_heads, dtype=args.dtype
        )
        ring_global_lses_transposed = [
            lse.transpose(0, 1) for lse in ring_global_lses
        ]
        put_back_local_zigzag(
            ring_global_lses_transposed,
            ring_global_lse,
            ring_cu_padded_seqlens,
            dist.get_world_size(),
        )
        ring_global_lse = ring_global_lse.transpose(0, 1)
    else:
        ring_global_lse = None

    ring_global_out, ring_global_lse = unpad_attention_output(
        ring_global_out, ring_global_lse, raw_seqlens, ring_padded_seqlens
    )

    if not args.skip_backward:
        ring_global_dqs = [
            torch.zeros_like(ring_local_dq)
            for _ in range(dist.get_world_size())
        ]
        dist.all_gather(ring_global_dqs, ring_local_dq)
        ring_global_dq = torch.zeros(
            sum(ring_padded_seqlens),
            args.n_heads,
            args.head_dim,
            dtype=args.dtype,
        )
        put_back_local_zigzag(
            ring_global_dqs,
            ring_global_dq,
            ring_cu_padded_seqlens,
            dist.get_world_size(),
        )

        ring_global_dkvs = [
            torch.zeros_like(ring_local_dkv)
            for _ in range(dist.get_world_size())
        ]
        dist.all_gather(ring_global_dkvs, ring_local_dkv)
        ring_global_dkv = torch.zeros(
            sum(ring_padded_seqlens),
            2,
            args.n_query_groups,
            args.head_dim,
            dtype=args.dtype,
        )
        put_back_local_zigzag(
            ring_global_dkvs,
            ring_global_dkv,
            ring_cu_padded_seqlens,
            dist.get_world_size(),
        )

        ring_global_dq, ring_global_dkv = unpad_attention_grad(
            ring_global_dq, ring_global_dkv, raw_seqlens, ring_padded_seqlens
        )

    # split output by seqlens
    cumsum_seqlens = np.cumsum([0] + raw_seqlens)
    per_seq_out = [
        out[cumsum_seqlens[i] : cumsum_seqlens[i + 1]]
        for i in range(len(raw_seqlens))
    ]
    if ref_lse is not None:
        per_seq_lse = [
            lse[:, cumsum_seqlens[i] : cumsum_seqlens[i + 1]]
            for i in range(len(raw_seqlens))
        ]
    per_seq_ref_out = [
        ref_out[cumsum_seqlens[i] : cumsum_seqlens[i + 1]]
        for i in range(len(raw_seqlens))
    ]
    if ref_lse is not None:
        per_seq_ref_lse = [
            ref_lse[:, cumsum_seqlens[i] : cumsum_seqlens[i + 1]]
            for i in range(len(raw_seqlens))
        ]
    else:
        per_seq_ref_lse = None
    per_seq_ring_out = [
        ring_global_out[cumsum_seqlens[i] : cumsum_seqlens[i + 1]]
        for i in range(len(raw_seqlens))
    ]
    if ring_global_lse is not None:
        per_seq_ring_lse = [
            ring_global_lse[:, cumsum_seqlens[i] : cumsum_seqlens[i + 1]]
            for i in range(len(raw_seqlens))
        ]
    else:
        per_seq_ring_lse = None
    if not args.skip_backward:
        per_seq_dq = [
            dq[cumsum_seqlens[i] : cumsum_seqlens[i + 1]]
            for i in range(len(raw_seqlens))
        ]
        per_seq_dkv = [
            dkv[cumsum_seqlens[i] : cumsum_seqlens[i + 1]]
            for i in range(len(raw_seqlens))
        ]
        per_seq_ref_dq = [
            ref_dq[cumsum_seqlens[i] : cumsum_seqlens[i + 1]]
            for i in range(len(raw_seqlens))
        ]
        per_seq_ref_dkv = [
            ref_dkv[cumsum_seqlens[i] : cumsum_seqlens[i + 1]]
            for i in range(len(raw_seqlens))
        ]
        per_seq_ring_dq = [
            ring_global_dq[cumsum_seqlens[i] : cumsum_seqlens[i + 1]]
            for i in range(len(raw_seqlens))
        ]
        per_seq_ring_dkv = [
            ring_global_dkv[cumsum_seqlens[i] : cumsum_seqlens[i + 1]]
            for i in range(len(raw_seqlens))
        ]
    colored_out = termcolor.colored("out", "green")
    colored_lse = termcolor.colored("lse", "blue")
    colored_dq = termcolor.colored("dq", "yellow")
    colored_dkv = termcolor.colored("dkv", "magenta")
    for i in range(len(raw_seqlens)):
        ref_out = per_seq_ref_out[i]
        if per_seq_ref_lse:
            ref_lse = per_seq_ref_lse[i]
        else:
            ref_lse = None
        out = per_seq_out[i]
        if ref_lse is not None:
            lse = per_seq_lse[i]
        else:
            lse = None
        ring_out = per_seq_ring_out[i]
        if per_seq_ring_lse:
            ring_lse = per_seq_ring_lse[i]
        else:
            ring_lse = None
        if not args.skip_backward:
            ref_dq = per_seq_ref_dq[i]
            ref_dkv = per_seq_ref_dkv[i]
            dq = per_seq_dq[i]
            dkv = per_seq_dkv[i]
            ring_dq = per_seq_ring_dq[i]
            ring_dkv = per_seq_ring_dkv[i]
        if ref_lse is not None and ring_lse is not None:
            lse_mean_diff = (ref_lse - lse).abs().mean()
            lse_max_diff = (ref_lse - lse).abs().max()
            ring_lse_mean_diff = (ref_lse - ring_lse).abs().mean()
            ring_lse_max_diff = (ref_lse - ring_lse).abs().max()
        # if dist.get_rank() == 0:
        #     import code
        #     code.interact(local=locals())
        # dist.barrier(device_ids=[torch.cuda.current_device()])
        print_0(f"Seq {i}:")
        print_0(
            f"\t{colored_out} mean diff: dcp: {(out - ref_out).abs().mean()}, ring: {(ring_out - ref_out).abs().mean()}"
        )
        print_0(
            f"\t{colored_out} max diff: dcp: {(out - ref_out).abs().max()}, ring: {(ring_out - ref_out).abs().max()}"
        )
        if ref_lse is not None and ring_lse is not None:
            print_0(
                f"\t{colored_lse} mean diff: dcp: {lse_mean_diff}, ring: {ring_lse_mean_diff}"
            )
            print_0(
                f"\t{colored_lse} max diff: dcp: {lse_max_diff}, ring: {ring_lse_max_diff}"
            )
        # print lse diff per block
        if ref_lse is not None and ring_lse is not None:
            n_blocks = raw_seqlens[i] // args.block_size
            s = f"\t{colored_lse} diff per block:\n"
            for head in range(args.n_heads):
                s += f"\t\tHead{head}"
                for j in range(n_blocks):
                    block_lse = lse[
                        head, j * args.block_size : (j + 1) * args.block_size
                    ]
                    ref_block_lse = ref_lse[
                        head, j * args.block_size : (j + 1) * args.block_size
                    ]
                    block_lse_mean_diff = (
                        (ref_block_lse - block_lse).abs().mean()
                    )
                    s += f" {block_lse_mean_diff:.4f}"
                s += "\n"
            print_0(s)
        if not args.skip_backward:
            print_0(
                f"\t{colored_dq} mean diff: dcp: {(dq - ref_dq).abs().mean()}, ring: {(ring_dq - ref_dq).abs().mean()}"
            )
            print_0(
                f"\t{colored_dq} max diff: dcp: {(dq - ref_dq).abs().max()}, ring: {(ring_dq - ref_dq).abs().max()}"
            )
            print_0(
                f"\t{colored_dkv} mean diff: dcp: {(dkv - ref_dkv).abs().mean()}, ring: {(ring_dkv - ref_dkv).abs().mean()}"
            )
            print_0(
                f"\t{colored_dkv} max diff: dcp: {(dkv - ref_dkv).abs().max()}, ring: {(ring_dkv - ref_dkv).abs().max()}"
            )
    dist.destroy_process_group()
