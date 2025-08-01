import pickle
from typing import Dict, List, Tuple, Optional

import numpy as np

import torch
import torch.distributed as dist
import torch.nn.functional as F

from einops import rearrange, repeat

from dcp.core.block_table import BlockType, WorkloadSpec
from dcp.core.instructions import ExecutionPlan
from dcp.runtime.flash_attention.executor import flash_attn_2_5_8_plus

if not flash_attn_2_5_8_plus:
    try:
        from ring_flash_attn.triton_utils import unflatten_varlen_lse
    except:
        from ring_flash_attn.utils import unflatten_varlen_lse


def get_byte_tensor(b: bytes):
    return torch.tensor(
        list(b), dtype=torch.uint8, device=torch.get_default_device()
    )


def read_byte_from_tensor(t: torch.Tensor):
    return bytes(t.cpu().tolist())


def reconstruct_attention_forward_output_for_test(
    local_out: torch.Tensor,
    local_lse: torch.Tensor,
    q_shape: Tuple[int, ...],
    qkv_dtype: torch.dtype,
    cumsum_seqlens: torch.Tensor,
    workload_spec: WorkloadSpec,
    exec_plan: ExecutionPlan,
    n_devices_per_node: int,
):
    dist_rank = dist.get_rank()
    # allgather output and lse buffers
    local_out_and_lse_shapes = (
        local_out.shape,
        local_out.dtype,
        local_lse.shape,
        local_lse.dtype,
    )
    pickled_shapes = pickle.dumps(local_out_and_lse_shapes)
    # first send the size of the pickled object
    pickled_size = torch.tensor([len(pickled_shapes)], dtype=torch.int64)
    all_sizes = [
        torch.zeros_like(pickled_size) for _ in range(dist.get_world_size())
    ]
    dist.all_gather(all_sizes, pickled_size)
    all_sizes = [s.item() for s in all_sizes]
    global_shapes = [torch.zeros(s, dtype=torch.uint8) for s in all_sizes]
    dist.all_gather(global_shapes, get_byte_tensor(pickled_shapes))
    global_shapes = [
        pickle.loads(read_byte_from_tensor(gs)) for gs in global_shapes
    ]
    global_out_and_lses = []
    for dev_rank, shapes in enumerate(global_shapes):
        out_shape, out_dtype, lse_shape, lse_dtype = shapes
        # broadcast tensors
        res_out = torch.empty(out_shape, dtype=out_dtype)
        res_lse = torch.empty(lse_shape, dtype=lse_dtype)
        if dev_rank == dist_rank:
            dist.broadcast(local_out, src=dev_rank)
            dist.broadcast(local_lse, src=dev_rank)
            global_out_and_lses.append((local_out, local_lse))
        else:
            dist.broadcast(res_out, src=dev_rank)
            dist.broadcast(res_lse, src=dev_rank)
            global_out_and_lses.append((res_out, res_lse))
    # allgather local_cu_seqlens from exec_plan
    local_cu_seqlens = torch.tensor(
        exec_plan.local_cu_seqlens, dtype=torch.int32
    )
    # first gather the shape of local_cu_seqlens
    cu_seqlens_shape = torch.tensor(
        [local_cu_seqlens.shape[0]], dtype=torch.int64
    )
    all_cu_seqlens_shapes = [
        torch.zeros_like(cu_seqlens_shape)
        for _ in range(dist.get_world_size())
    ]
    dist.all_gather(all_cu_seqlens_shapes, cu_seqlens_shape)
    all_cu_seqlens = [
        torch.zeros(all_cu_seqlens_shapes[i].item(), dtype=torch.int32)
        for i in range(dist.get_world_size())
    ]
    dist.all_gather(all_cu_seqlens, local_cu_seqlens)
    # reconstruct output tensors from global output and lse buffers
    global_qkv_out = torch.empty(
        (cumsum_seqlens[-1], q_shape[-2], q_shape[-1]), dtype=qkv_dtype
    )
    global_lse_out = torch.empty(
        (q_shape[1], cumsum_seqlens[-1]), dtype=torch.float32
    )
    for output_id in range(len(workload_spec.output_sizes)):
        out_meta = workload_spec.block_mapping.output_id_to_meta
        seq_id = out_meta[output_id].seq_id
        block_idx = out_meta[output_id].block_id
        block_size = out_meta[output_id].block_size
        head_offset = (
            out_meta[output_id].head_id * out_meta[output_id].head_block_size
        )
        head_block_size = out_meta[output_id].head_block_size
        n_tokens = out_meta[output_id].n_tokens
        seq_offset = cumsum_seqlens[seq_id].item()
        block_start = seq_offset + block_idx * block_size
        # get corresponding output from global_out_and_lses
        out_device = workload_spec.output_to_device_map[output_id]
        out_rank = out_device[0] * n_devices_per_node + out_device[1]
        out_tensor, lse_tensor = global_out_and_lses[out_rank]
        out_block_index = (
            workload_spec.block_mapping.output_id_to_buffer_index[out_device][
                output_id
            ]
        )
        local_cu_seqlens = all_cu_seqlens[out_rank]
        out_token_offset = local_cu_seqlens[out_block_index].item()
        if out_meta[output_id].type == BlockType.Out:
            actual_out = out_tensor[
                out_token_offset : out_token_offset + n_tokens
            ]
            x = global_qkv_out[
                block_start : block_start + n_tokens,
                head_offset : head_offset + head_block_size,
            ].shape
            y = actual_out.shape
            global_qkv_out[
                block_start : block_start + n_tokens,
                head_offset : head_offset + head_block_size,
            ] = actual_out
        elif out_meta[output_id].type == BlockType.LSE:
            # lse buffer is of shape (n_heads, sum(local_seqlens))
            actual_lse = lse_tensor[
                :, out_token_offset : out_token_offset + n_tokens
            ]
            global_lse_out[
                head_offset : head_offset + head_block_size,
                block_start : block_start + n_tokens,
            ] = actual_lse
        else:
            raise ValueError(
                "Unknown output block type: {}".format(
                    out_meta[output_id].type
                )
            )

    if not flash_attn_2_5_8_plus:
        max_seqlen = (cumsum_seqlens[1:] - cumsum_seqlens[:-1]).max().item()
        global_lse_out = unflatten_varlen_lse(
            global_lse_out.transpose(-2, -1).unsqueeze(-1),
            cumsum_seqlens,
            max_seqlen,
        )
    return global_qkv_out, global_lse_out


def reconstruct_attention_backward_output_for_test(
    local_dq: torch.Tensor,
    local_dkv: torch.Tensor,
    q: torch.Tensor,
    kv: torch.Tensor,
    bw_workload: WorkloadSpec,
    bw_exec_plan: ExecutionPlan,
    cumsum_seqlens: torch.Tensor,
    n_devices_per_node: int,
):
    dist_rank = dist.get_rank()
    # allgather output and lse buffers
    local_out_and_lse_shapes = (
        local_dq.shape,
        local_dq.dtype,
        local_dkv.shape,
        local_dkv.dtype,
    )
    pickled_shapes = pickle.dumps(local_out_and_lse_shapes)
    # first send the size of the pickled object
    pickled_size = torch.tensor([len(pickled_shapes)], dtype=torch.int64)
    all_sizes = [
        torch.zeros_like(pickled_size) for _ in range(dist.get_world_size())
    ]
    dist.all_gather(all_sizes, pickled_size)
    all_sizes = [s.item() for s in all_sizes]
    global_shapes = [torch.zeros(s, dtype=torch.uint8) for s in all_sizes]
    dist.all_gather(global_shapes, get_byte_tensor(pickled_shapes))
    global_shapes = [
        pickle.loads(read_byte_from_tensor(gs)) for gs in global_shapes
    ]
    global_dqs_and_dkvs = []
    for dev_rank, shapes in enumerate(global_shapes):
        dq_shape, dq_dtype, dkv_shape, dkv_dtype = shapes
        # broadcast tensors
        res_dq = torch.empty(dq_shape, dtype=dq_dtype)
        res_dkv = torch.empty(dkv_shape, dtype=dkv_dtype)
        if dev_rank == dist_rank:
            dist.broadcast(local_dq, src=dev_rank)
            dist.broadcast(local_dkv, src=dev_rank)
            global_dqs_and_dkvs.append((local_dq, local_dkv))
        else:
            dist.broadcast(res_dq, src=dev_rank)
            dist.broadcast(res_dkv, src=dev_rank)
            global_dqs_and_dkvs.append((res_dq, res_dkv))
    # allgather local_cu_seqlens from exec_plan
    local_cu_seqlens = torch.tensor(
        bw_exec_plan.local_cu_seqlens, dtype=torch.int32
    )
    # first gather the shape of local_cu_seqlens
    cu_seqlens_shape = torch.tensor(
        [local_cu_seqlens.shape[0]], dtype=torch.int64
    )
    all_cu_seqlens_shapes = [
        torch.zeros_like(cu_seqlens_shape)
        for _ in range(dist.get_world_size())
    ]
    dist.all_gather(all_cu_seqlens_shapes, cu_seqlens_shape)
    all_cu_seqlens = [
        torch.zeros(all_cu_seqlens_shapes[i].item(), dtype=torch.int32)
        for i in range(dist.get_world_size())
    ]
    dist.all_gather(all_cu_seqlens, local_cu_seqlens)
    # reconstruct output tensors from global output and lse buffers
    global_dq_out = torch.empty(
        (cumsum_seqlens[-1], q.shape[-2], q.shape[-1]), dtype=q.dtype
    )
    global_dkv_out = torch.empty(
        (cumsum_seqlens[-1], 2, kv.shape[2], kv.shape[3]), dtype=kv.dtype
    )
    for output_id in range(len(bw_workload.output_sizes)):
        out_meta = bw_workload.block_mapping.output_id_to_meta
        seq_id = out_meta[output_id].seq_id
        block_idx = out_meta[output_id].block_id
        block_size = out_meta[output_id].block_size
        head_offset = (
            out_meta[output_id].head_id * out_meta[output_id].head_block_size
        )
        head_block_size = out_meta[output_id].head_block_size
        n_tokens = out_meta[output_id].n_tokens
        seq_offset = cumsum_seqlens[seq_id].item()
        block_start = seq_offset + block_idx * block_size
        # get corresponding output from global_out_and_lses
        out_device = bw_workload.output_to_device_map[output_id]
        out_rank = out_device[0] * n_devices_per_node + out_device[1]
        dq_tensor, dkv_tensor = global_dqs_and_dkvs[out_rank]
        dq_block_index = bw_workload.block_mapping.output_id_to_buffer_index[
            out_device
        ][output_id]
        local_cu_seqlens = all_cu_seqlens[out_rank]
        dq_token_offset = local_cu_seqlens[dq_block_index].item()
        if out_meta[output_id].type == BlockType.dQ:
            actual_dq = dq_tensor[dq_token_offset : dq_token_offset + n_tokens]
            global_dq_out[
                block_start : block_start + n_tokens,
                head_offset : head_offset + head_block_size,
            ] = actual_dq
        elif out_meta[output_id].type == BlockType.dKV:
            actual_dkv = dkv_tensor[
                dq_token_offset : dq_token_offset + n_tokens
            ]
            global_dkv_out[
                block_start : block_start + n_tokens,
                :,
                head_offset : head_offset + head_block_size,
            ] = actual_dkv
        else:
            raise ValueError(
                "Unknown output block type: {}".format(
                    out_meta[output_id].type
                )
            )
    return global_dq_out, global_dkv_out


def pad_q_kv_dout_to_max_seqlen(
    q: torch.Tensor,
    kv: torch.Tensor,
    dout: torch.Tensor,
    seqlens: List[int],
    max_seqlen: int,
):
    padded_q = torch.zeros(
        (len(seqlens), max_seqlen, q.shape[-2], q.shape[-1]),
        dtype=q.dtype,
    )
    padded_kv = torch.zeros(
        (len(seqlens), max_seqlen, 2, kv.shape[-2], kv.shape[-1]),
        dtype=kv.dtype,
    )
    padded_dout = torch.zeros(
        (len(seqlens), max_seqlen, dout.shape[1], dout.shape[2]),
        dtype=dout.dtype,
    )

    padding_mask = torch.ones((len(seqlens), max_seqlen), dtype=torch.bool)
    cu_seqlens = np.cumsum([0] + seqlens).tolist()
    for i, seqlen in enumerate(seqlens):
        padded_q[i, :seqlen] = q[cu_seqlens[i] : cu_seqlens[i] + seqlen]
        padded_kv[i, :seqlen] = kv[cu_seqlens[i] : cu_seqlens[i] + seqlen]
        padded_dout[i, :seqlen] = dout[cu_seqlens[i] : cu_seqlens[i] + seqlen]
        padding_mask[i, seqlen:] = False
    return padded_q, padded_kv, padded_dout, padding_mask


def pad_q_kv(
    q: torch.Tensor,
    kv: torch.Tensor,
    raw_seqlens: List[int],
    padded_seqlens: List[int],
):
    padded_q = torch.zeros(
        (sum(padded_seqlens), q.shape[-2], q.shape[-1]),
        dtype=q.dtype,
    )
    padded_kv = torch.zeros(
        (sum(padded_seqlens), 2, kv.shape[2], kv.shape[3]),
        dtype=kv.dtype,
    )
    cu_raw_seqlens = np.cumsum([0] + raw_seqlens).tolist()
    cu_padded_seqlens = np.cumsum([0] + padded_seqlens).tolist()
    for i in range(len(raw_seqlens)):
        padded_q[
            cu_padded_seqlens[i] : cu_padded_seqlens[i] + raw_seqlens[i]
        ] = q[cu_raw_seqlens[i] : cu_raw_seqlens[i] + raw_seqlens[i]]
        padded_kv[
            cu_padded_seqlens[i] : cu_padded_seqlens[i] + raw_seqlens[i]
        ] = kv[cu_raw_seqlens[i] : cu_raw_seqlens[i] + raw_seqlens[i]]
    return padded_q, padded_kv


def unpad_q_kv(
    q: torch.Tensor,
    kv: torch.Tensor,
    raw_seqlens: List[int],
    padded_seqlens: List[int],
):
    unpadded_q = torch.empty(
        (sum(raw_seqlens), q.shape[-2], q.shape[-1]),
        dtype=q.dtype,
    )
    unpadded_kv = torch.empty(
        (sum(raw_seqlens), 2, kv.shape[2], kv.shape[3]),
        dtype=kv.dtype,
    )
    cu_raw_seqlens = np.cumsum([0] + raw_seqlens).tolist()
    cu_padded_seqlens = np.cumsum([0] + padded_seqlens).tolist()
    for i in range(len(raw_seqlens)):
        unpadded_q[cu_raw_seqlens[i] : cu_raw_seqlens[i + 1]] = q[
            cu_padded_seqlens[i] : cu_padded_seqlens[i] + raw_seqlens[i]
        ]
        unpadded_kv[cu_raw_seqlens[i] : cu_raw_seqlens[i + 1]] = kv[
            cu_padded_seqlens[i] : cu_padded_seqlens[i] + raw_seqlens[i]
        ]
    return unpadded_q, unpadded_kv


def pad_dout(
    dout: torch.Tensor,
    raw_seqlens: List[int],
    padded_seqlens: List[int],
):
    padded_dout = torch.zeros(
        (sum(padded_seqlens), dout.shape[1], dout.shape[2]),
        dtype=dout.dtype,
    )
    cu_raw_seqlens = np.cumsum([0] + raw_seqlens).tolist()
    cu_padded_seqlens = np.cumsum([0] + padded_seqlens).tolist()
    for i in range(len(raw_seqlens)):
        padded_dout[
            cu_padded_seqlens[i] : cu_padded_seqlens[i] + raw_seqlens[i]
        ] = dout[cu_raw_seqlens[i] : cu_raw_seqlens[i] + raw_seqlens[i]]
    return padded_dout


def unpad_attention_output(
    qkv_out: torch.Tensor,
    lse_out: torch.Tensor,
    raw_seqlens: List[int],
    padded_seqlens: List[int],
):
    unpadded_out = torch.empty(
        (sum(raw_seqlens), qkv_out.shape[1], qkv_out.shape[2]),
        dtype=qkv_out.dtype,
    )
    if lse_out is not None:
        unpadded_lse = torch.empty(
            (lse_out.shape[0], sum(raw_seqlens)), dtype=torch.float32
        )
    else:
        unpadded_lse = None
    cu_raw_seqlens = np.cumsum([0] + raw_seqlens).tolist()
    cu_padded_seqlens = np.cumsum([0] + padded_seqlens).tolist()
    for i in range(len(raw_seqlens)):
        a = unpadded_out[cu_raw_seqlens[i] : cu_raw_seqlens[i + 1]].shape
        b = qkv_out[
            cu_padded_seqlens[i] : cu_padded_seqlens[i] + raw_seqlens[i]
        ].shape
        unpadded_out[cu_raw_seqlens[i] : cu_raw_seqlens[i + 1]] = qkv_out[
            cu_padded_seqlens[i] : cu_padded_seqlens[i] + raw_seqlens[i]
        ]
        if lse_out is not None:
            unpadded_lse[:, cu_raw_seqlens[i] : cu_raw_seqlens[i + 1]] = (
                lse_out[
                    :,
                    cu_padded_seqlens[i] : cu_padded_seqlens[i]
                    + raw_seqlens[i],
                ]
            )
    return unpadded_out, unpadded_lse


def unpad_attention_grad(
    dq: torch.Tensor,
    dkv: torch.Tensor,
    raw_seqlens: List[int],
    padded_seqlens: List[int],
):
    unpadded_dq = torch.empty(
        (sum(raw_seqlens), dq.shape[1], dq.shape[2]),
        dtype=dq.dtype,
    )
    unpadded_dkv = torch.empty(
        (sum(raw_seqlens), 2, dkv.shape[2], dkv.shape[3]),
        dtype=dkv.dtype,
    )
    cu_raw_seqlens = np.cumsum([0] + raw_seqlens).tolist()
    cu_padded_seqlens = np.cumsum([0] + padded_seqlens).tolist()
    for i in range(len(raw_seqlens)):
        unpadded_dq[cu_raw_seqlens[i] : cu_raw_seqlens[i + 1]] = dq[
            cu_padded_seqlens[i] : cu_padded_seqlens[i] + raw_seqlens[i]
        ]
        unpadded_dkv[cu_raw_seqlens[i] : cu_raw_seqlens[i + 1]] = dkv[
            cu_padded_seqlens[i] : cu_padded_seqlens[i] + raw_seqlens[i]
        ]
    return unpadded_dq, unpadded_dkv


# from FlashAttn
# flash_attn/bert_padding.py
class IndexFirstAxis(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, indices):
        ctx.save_for_backward(indices)
        assert input.ndim >= 2
        ctx.first_axis_dim, other_shape = input.shape[0], input.shape[1:]
        second_dim = other_shape.numel()
        # TD [2022-03-04] For some reason torch.gather is a bit faster than indexing.
        # return input[indices]
        return torch.gather(
            rearrange(input, "b ... -> b (...)"),
            0,
            repeat(indices, "z -> z d", d=second_dim),
        ).reshape(-1, *other_shape)

    @staticmethod
    def backward(ctx, grad_output):
        (indices,) = ctx.saved_tensors
        assert grad_output.ndim >= 2
        other_shape = grad_output.shape[1:]
        grad_output = rearrange(grad_output, "b ... -> b (...)")
        grad_input = torch.zeros(
            [ctx.first_axis_dim, grad_output.shape[1]],
            device=grad_output.device,
            dtype=grad_output.dtype,
        )
        # TD [2022-03-04] For some reason torch.scatter is a bit faster than indexing.
        # grad_input[indices] = grad_output
        grad_input.scatter_(
            0, repeat(indices, "z -> z d", d=grad_output.shape[1]), grad_output
        )
        return grad_input.reshape(ctx.first_axis_dim, *other_shape), None


index_first_axis = IndexFirstAxis.apply


def unpad_input(hidden_states: torch.Tensor, padding_mask: torch.Tensor):
    """
    Arguments:
        hidden_states: (batch, seqlen, ...)
        padding_mask: (batch, seqlen)
    Return:
        hidden_states: (total_nnz, ...), where total_nnz = number of tokens selected in padding_mask
    """
    indices = torch.nonzero(padding_mask.flatten(), as_tuple=False).flatten()
    # TD [2022-03-04] We don't want to index with a bool mask, because Pytorch will expand the
    # bool mask, then call nonzero to get the indices, then index with those. The indices is @dim
    # times larger than it needs to be, wasting memory. It's faster and more memory-efficient to
    # index with integer indices. Moreover, torch's index is a bit slower than it needs to be,
    # so we write custom forward and backward to make it a bit faster.
    return index_first_axis(
        rearrange(hidden_states, "b s ... -> (b s) ..."), indices
    )


def lambda_mask_fn(
    raw_seqlens: List[int],
    padded_seqlens: List[int],
    mask_dtype: torch.dtype,
    mask_fn_info: Optional[List[Dict]] = None,
    n_starting: int = 64,
    l_pretrain: int = 4096,
):
    attn_mask = torch.zeros((sum(padded_seqlens), 2, 2), dtype=mask_dtype)
    cu_seqlens_padded = F.pad(
        torch.cumsum(torch.tensor(padded_seqlens, dtype=mask_dtype), 0), (1, 0)
    ).to(mask_dtype)
    for seq_id, seqlen in enumerate(raw_seqlens):
        for i in range(seqlen):
            # first range, 0 to n_starting
            range1_start = 0
            range1_end = min(n_starting, i + 1)
            # second range, windowed with l_pretrain
            range2_start = max(0, i - l_pretrain + 1)
            range2_end = i + 1
            attn_mask[cu_seqlens_padded[seq_id] + i, 0, 0] = range1_start
            attn_mask[cu_seqlens_padded[seq_id] + i, 0, 1] = range1_end
            attn_mask[cu_seqlens_padded[seq_id] + i, 1, 0] = range2_start
            attn_mask[cu_seqlens_padded[seq_id] + i, 1, 1] = range2_end
    return attn_mask


def causal_range_mask_fn(
    raw_seqlens: List[int],
    padded_seqlens: List[int],
    mask_dtype: torch.dtype,
    mask_fn_info: Optional[List[Dict]] = None,
):
    attn_mask = torch.zeros((sum(padded_seqlens), 2), dtype=mask_dtype)
    cu_seqlens_padded = F.pad(
        torch.cumsum(torch.tensor(padded_seqlens, dtype=mask_dtype), 0), (1, 0)
    ).to(mask_dtype)
    for seq_id, seqlen in enumerate(raw_seqlens):
        for i in range(seqlen):
            attn_mask[cu_seqlens_padded[seq_id] + i, 0] = 0
            attn_mask[cu_seqlens_padded[seq_id] + i, 1] = i + 1
    return attn_mask


def shared_question_mask_fn(
    raw_seqlens: List[int],
    padded_seqlens: List[int],
    mask_dtype: torch.dtype,
    mask_fn_info: Optional[List[Dict]] = None,
    question_proportion: float = 0.2,
    n_answers: int = 4,
):
    attn_mask = torch.zeros((sum(padded_seqlens), 2, 2), dtype=mask_dtype)
    cu_seqlens_padded = F.pad(
        torch.cumsum(torch.tensor(padded_seqlens, dtype=mask_dtype), 0), (1, 0)
    ).to(mask_dtype)
    for seq_id, seqlen in enumerate(raw_seqlens):
        n_question_tokens = int(seqlen * question_proportion)
        n_answer_tokens = seqlen - n_question_tokens
        per_answer_tokens = n_answer_tokens // n_answers
        for i in range(seqlen):
            if i < n_question_tokens:
                # normal causal mask
                attn_mask[cu_seqlens_padded[seq_id] + i, 0, 0] = 0
                attn_mask[cu_seqlens_padded[seq_id] + i, 0, 1] = i + 1
                attn_mask[cu_seqlens_padded[seq_id] + i, 1, 0] = i + 1
                attn_mask[cu_seqlens_padded[seq_id] + i, 1, 1] = i + 1
            else:
                # full mask on question tokens, causal mask on current answer
                question_start = 0
                question_end = n_question_tokens
                if (
                    i
                    < (per_answer_tokens * (n_answers - 1)) + n_question_tokens
                ):
                    answer_start = (
                        (i - n_question_tokens)
                        // per_answer_tokens
                        * per_answer_tokens
                        + n_question_tokens
                    )
                else:
                    # last answer
                    answer_start = (
                        n_question_tokens + (n_answers - 1) * per_answer_tokens
                    )
                answer_end = i + 1
                attn_mask[cu_seqlens_padded[seq_id] + i, 0, 0] = question_start
                attn_mask[cu_seqlens_padded[seq_id] + i, 0, 1] = question_end
                attn_mask[cu_seqlens_padded[seq_id] + i, 1, 0] = answer_start
                attn_mask[cu_seqlens_padded[seq_id] + i, 1, 1] = answer_end
    return attn_mask


def causal_blockwise_mask_fn(
    raw_seqlens: List[int],
    padded_seqlens: List[int],
    mask_dtype: torch.dtype,
    mask_fn_info: Optional[List[Dict]] = None,
    block_size: int = 256,
    k_local: int = 2,
):
    attn_mask = torch.zeros((sum(padded_seqlens), 2, 2), dtype=mask_dtype)
    cu_seqlens_padded = F.pad(
        torch.cumsum(torch.tensor(padded_seqlens, dtype=mask_dtype), 0), (1, 0)
    ).to(mask_dtype)
    for seq_id, seqlen in enumerate(raw_seqlens):
        for i in range(seqlen):
            block_id = i // block_size
            n_blocks = (seqlen + block_size - 1) // block_size
            if block_id == 0:
                # first block, causal
                attn_mask[cu_seqlens_padded[seq_id] + i, 0, 0] = 0
                attn_mask[cu_seqlens_padded[seq_id] + i, 0, 1] = i + 1
                attn_mask[cu_seqlens_padded[seq_id] + i, 1, 0] = i + 1
                attn_mask[cu_seqlens_padded[seq_id] + i, 1, 1] = i + 1
            elif block_id == n_blocks - 1:
                # attend to all previous blocks
                prev_block_end = (block_id - 1) * block_size
                attn_mask[cu_seqlens_padded[seq_id] + i, 0, 0] = 0
                attn_mask[cu_seqlens_padded[seq_id] + i, 0, 1] = prev_block_end
                # causal within this block
                attn_mask[cu_seqlens_padded[seq_id] + i, 1, 0] = prev_block_end
                attn_mask[cu_seqlens_padded[seq_id] + i, 1, 1] = i + 1
            else:
                # attend to first block
                attn_mask[cu_seqlens_padded[seq_id] + i, 0, 0] = 0
                attn_mask[cu_seqlens_padded[seq_id] + i, 0, 1] = block_size
                # attend to previous k_local blocks
                prev_block_end = max(block_size, i - k_local * block_size)
                attn_mask[cu_seqlens_padded[seq_id] + i, 1, 0] = prev_block_end
                attn_mask[cu_seqlens_padded[seq_id] + i, 1, 1] = i + 1
    return attn_mask


def modality_specific_mask_fn(
    raw_seqlens: List[int],
    padded_seqlens: List[int],
    mask_dtype: torch.dtype,
    mask_fn_info: Optional[List[Dict]] = None,
    limit_image_attn_to_self: bool = False,
):
    # for text tokens, attend to all previous tokens (i.e., causal)
    # for image tokens,
    # if limit_image_attn_to_self is True,attend only to other tokens
    # within the image (bidirectional). Otherwise, apply bidirectional
    # attention within the image and causal attention to all previous tokens
    assert mask_fn_info is not None
    assert len(mask_fn_info) == len(raw_seqlens)
    attn_mask = torch.zeros((sum(padded_seqlens), 2), dtype=mask_dtype)
    cu_seqlens_padded = F.pad(
        torch.cumsum(torch.tensor(padded_seqlens, dtype=mask_dtype), 0), (1, 0)
    ).to(mask_dtype)
    for seq_id, seqlen in enumerate(raw_seqlens):
        mask_info_dict = mask_fn_info[seq_id]
        assert "image_size" in mask_info_dict
        assert "text_lengths" in mask_info_dict
        image_size = mask_info_dict["image_size"]
        text_lengths = mask_info_dict["text_lengths"]
        curr_seqlen = 0
        for idx, text_len in enumerate(text_lengths):
            for i in range(text_len):
                attn_mask[cu_seqlens_padded[seq_id] + curr_seqlen + i, 0] = 0
                attn_mask[cu_seqlens_padded[seq_id] + curr_seqlen + i, 1] = (
                    curr_seqlen + i + 1
                )
            curr_seqlen += text_len
            if idx < len(text_lengths) - 1:
                # add image
                for i in range(image_size):
                    if limit_image_attn_to_self:
                        attn_mask[
                            cu_seqlens_padded[seq_id] + curr_seqlen + i, 0
                        ] = curr_seqlen
                    else:
                        attn_mask[
                            cu_seqlens_padded[seq_id] + curr_seqlen + i, 0
                        ] = 0
                    attn_mask[
                        cu_seqlens_padded[seq_id] + curr_seqlen + i, 1
                    ] = (curr_seqlen + image_size)
                curr_seqlen += image_size
    return attn_mask


def get_local_attn_ranges_zigzag(
    global_attn_ranges, padded_seqlens, world_size, rank
):
    padded_cu_seqlens = F.pad(
        torch.cumsum(torch.tensor(padded_seqlens, dtype=torch.int32), 0),
        (1, 0),
    ).to(torch.int32)
    if global_attn_ranges.dim() == 2:
        local_attn_ranges = torch.zeros(
            (2, sum(padded_seqlens) // world_size), dtype=torch.int32
        )
    else:
        local_attn_ranges = torch.zeros(
            (2, 2, sum(padded_seqlens) // world_size), dtype=torch.int32
        )
    for seq_id, seqlen in enumerate(padded_seqlens):
        local_chunk_size = seqlen // (2 * world_size)
        seq_offset = padded_cu_seqlens[seq_id]
        local_seq_offset = seq_offset // world_size
        first_chunk_start = rank * local_chunk_size + seq_offset
        second_chunk_start = (
            2 * world_size - rank - 1
        ) * local_chunk_size + seq_offset
        local_first_chunk_start = local_seq_offset
        local_second_chunk_start = local_seq_offset + local_chunk_size
        if local_attn_ranges.dim() == 2:
            local_attn_ranges[
                :,
                local_first_chunk_start : local_first_chunk_start
                + local_chunk_size,
            ] = global_attn_ranges[
                :, first_chunk_start : first_chunk_start + local_chunk_size
            ]
            local_attn_ranges[
                :,
                local_second_chunk_start : local_second_chunk_start
                + local_chunk_size,
            ] = global_attn_ranges[
                :, second_chunk_start : second_chunk_start + local_chunk_size
            ]
        else:
            local_attn_ranges[
                :,
                :,
                local_first_chunk_start : local_first_chunk_start
                + local_chunk_size,
            ] = global_attn_ranges[
                :, :, first_chunk_start : first_chunk_start + local_chunk_size
            ]
            local_attn_ranges[
                :,
                :,
                local_second_chunk_start : local_second_chunk_start
                + local_chunk_size,
            ] = global_attn_ranges[
                :,
                :,
                second_chunk_start : second_chunk_start + local_chunk_size,
            ]
    local_attn_ranges = local_attn_ranges.contiguous()
    return local_attn_ranges


def get_per_step_attn_range(
    step_idx: int,
    p2p_rank: int,
    cp_size: int,
    attn_ranges_cpu: torch.Tensor,
    cu_seqlens_cpu: torch.Tensor,
) -> torch.Tensor:
    torch.cuda.nvtx.range_push("get_per_step_attn_range")
    n_chunks = 2 * cp_size
    current_step_kv_rank = (p2p_rank - step_idx + cp_size) % cp_size
    raw_seqlens = cu_seqlens_cpu[1:] - cu_seqlens_cpu[:-1]
    local_total_cu_seqlens = cu_seqlens_cpu // cp_size
    chunk_size_per_seq = raw_seqlens // n_chunks
    chunk_indices = [
        current_step_kv_rank,
        2 * cp_size - current_step_kv_rank - 1,
    ]
    chunk_0_starts = torch.zeros(
        attn_ranges_cpu.shape[-1], dtype=attn_ranges_cpu.dtype, device="cpu"
    )
    chunk_0_ends = torch.zeros(
        attn_ranges_cpu.shape[-1], dtype=attn_ranges_cpu.dtype, device="cpu"
    )
    chunk_1_starts = torch.zeros(
        attn_ranges_cpu.shape[-1], dtype=attn_ranges_cpu.dtype, device="cpu"
    )
    chunk_1_ends = torch.zeros(
        attn_ranges_cpu.shape[-1], dtype=attn_ranges_cpu.dtype, device="cpu"
    )
    for seq_id in range(raw_seqlens.shape[0]):
        local_tokens_start = local_total_cu_seqlens[seq_id]
        local_tokens_end = local_total_cu_seqlens[seq_id + 1]
        chunk_0_starts[local_tokens_start:local_tokens_end] = (
            chunk_size_per_seq[seq_id] * chunk_indices[0]
        )
        chunk_0_ends[local_tokens_start:local_tokens_end] = (
            chunk_0_starts[local_tokens_start:local_tokens_end]
            + chunk_size_per_seq[seq_id]
        )
        chunk_1_starts[local_tokens_start:local_tokens_end] = (
            chunk_size_per_seq[seq_id] * chunk_indices[1]
        )
        chunk_1_ends[local_tokens_start:local_tokens_end] = (
            chunk_1_starts[local_tokens_start:local_tokens_end]
            + chunk_size_per_seq[seq_id]
        )
    block_sizes = chunk_0_ends - chunk_0_starts
    local_attn_ranges = torch.full(
        (2, 2, attn_ranges_cpu.shape[-1]),
        -1,
        dtype=attn_ranges_cpu.dtype,
        device="cpu",
        # pin_memory=True,
    )
    if attn_ranges_cpu.dim() == 2:
        range_start_in_chunk0 = (
            torch.clamp(
                attn_ranges_cpu[0, :], min=chunk_0_starts, max=chunk_0_ends
            )
            - chunk_0_starts
        )
        range_end_in_chunk0 = (
            torch.clamp(
                attn_ranges_cpu[1, :], min=chunk_0_starts, max=chunk_0_ends
            )
            - chunk_0_starts
        )
        range_start_in_chunk1 = (
            torch.clamp(
                attn_ranges_cpu[0, :], min=chunk_1_starts, max=chunk_1_ends
            )
            - chunk_1_starts
            + block_sizes
        )
        range_end_in_chunk1 = (
            torch.clamp(
                attn_ranges_cpu[1, :], min=chunk_1_starts, max=chunk_1_ends
            )
            - chunk_1_starts
            + block_sizes
        )
        local_attn_ranges[0, 0, :] = range_start_in_chunk0
        local_attn_ranges[0, 1, :] = range_end_in_chunk0
        local_attn_ranges[1, 0, :] = range_start_in_chunk1
        local_attn_ranges[1, 1, :] = range_end_in_chunk1
    else:
        range0_start_in_chunk0 = (
            torch.clamp(
                attn_ranges_cpu[0, 0, :], min=chunk_0_starts, max=chunk_0_ends
            )
            - chunk_0_starts
        )
        range0_end_in_chunk0 = (
            torch.clamp(
                attn_ranges_cpu[0, 1, :], min=chunk_0_starts, max=chunk_0_ends
            )
            - chunk_0_starts
        )
        range0_start_in_chunk1 = (
            torch.clamp(
                attn_ranges_cpu[0, 0, :], min=chunk_1_starts, max=chunk_1_ends
            )
            - chunk_1_starts
            + block_sizes
        )
        range0_end_in_chunk1 = (
            torch.clamp(
                attn_ranges_cpu[0, 1, :], min=chunk_1_starts, max=chunk_1_ends
            )
            - chunk_1_starts
            + block_sizes
        )

        range1_start_in_chunk0 = (
            torch.clamp(
                attn_ranges_cpu[1, 0, :], min=chunk_0_starts, max=chunk_0_ends
            )
            - chunk_0_starts
        )
        range1_end_in_chunk0 = (
            torch.clamp(
                attn_ranges_cpu[1, 1, :], min=chunk_0_starts, max=chunk_0_ends
            )
            - chunk_0_starts
        )
        range1_start_in_chunk1 = (
            torch.clamp(
                attn_ranges_cpu[1, 0, :], min=chunk_1_starts, max=chunk_1_ends
            )
            - chunk_1_starts
            + block_sizes
        )
        range1_end_in_chunk1 = (
            torch.clamp(
                attn_ranges_cpu[1, 1, :], min=chunk_1_starts, max=chunk_1_ends
            )
            - chunk_1_starts
            + block_sizes
        )
        valid_range0_chunk0 = range0_start_in_chunk0 < range0_end_in_chunk0
        valid_range1_chunk0 = range1_start_in_chunk0 < range1_end_in_chunk0
        valid_range0_chunk1 = range0_start_in_chunk1 < range0_end_in_chunk1
        valid_range1_chunk1 = range1_start_in_chunk1 < range1_end_in_chunk1

        local_attn_ranges[0, 0, ~valid_range0_chunk0] = range0_start_in_chunk1[
            ~valid_range0_chunk0
        ]
        local_attn_ranges[0, 1, ~valid_range0_chunk0] = range0_end_in_chunk1[
            ~valid_range0_chunk0
        ]
        local_attn_ranges[0, 0, ~valid_range0_chunk1] = range0_start_in_chunk0[
            ~valid_range0_chunk1
        ]
        local_attn_ranges[0, 1, ~valid_range0_chunk1] = range0_end_in_chunk0[
            ~valid_range0_chunk1
        ]
        local_attn_ranges[0, 0, valid_range0_chunk0 & valid_range0_chunk1] = (
            range0_start_in_chunk0[valid_range0_chunk0 & valid_range0_chunk1]
        )
        local_attn_ranges[0, 1, valid_range0_chunk0 & valid_range0_chunk1] = (
            range0_end_in_chunk1[valid_range0_chunk0 & valid_range0_chunk1]
        )

        local_attn_ranges[1, 0, ~valid_range1_chunk0] = range1_start_in_chunk1[
            ~valid_range1_chunk0
        ]
        local_attn_ranges[1, 1, ~valid_range1_chunk0] = range1_end_in_chunk1[
            ~valid_range1_chunk0
        ]
        local_attn_ranges[1, 0, ~valid_range1_chunk1] = range1_start_in_chunk0[
            ~valid_range1_chunk1
        ]
        local_attn_ranges[1, 1, ~valid_range1_chunk1] = range1_end_in_chunk0[
            ~valid_range1_chunk1
        ]
        local_attn_ranges[1, 0, valid_range1_chunk0 & valid_range1_chunk1] = (
            range1_start_in_chunk0[valid_range1_chunk0 & valid_range1_chunk1]
        )
        local_attn_ranges[1, 1, valid_range1_chunk0 & valid_range1_chunk1] = (
            range1_end_in_chunk1[valid_range1_chunk0 & valid_range1_chunk1]
        )
    torch.cuda.nvtx.range_pop()
    return local_attn_ranges
