import triton
import triton.language as tl
from triton.compiler import make_backend
from triton.runtime.driver import driver
import torch
import torch.nn.functional as F
import numpy as np


@triton.jit
def _or_combine(a, b):
    return a or b


@triton.jit(do_not_specialize_on_alignment=["dst_lse_head_stride"])
def _flash_attn_update_out_lse(
    src_out_buffer,  # [src_n_total_blocks, block_size, n_head, head_dim]
    dst_out_buffer,  # [sum(n_tokens_per_block), n_head, head_dim]
    src_lse_buffer,  # [n_head, src_n_total_blocks, block_size]
    dst_lse_buffer,  # [n_head, sum(n_tokens_per_block)]
    n_head,  # number of attention heads
    src_block_table,  # (n_red_ops, MAX_N_BLOCKS_PER_OP)
    dst_token_offsets,  # (n_red_ops)
    n_tokens_per_block,  # (n_red_ops)
    block_lengths,  # length = n_red_ops, number of src blocks per op
    block_size,  # number of tokens in a block
    out_block_stride,  # = block_size * n_head * head_dim
    out_token_stride,  # = n_head * head_dim
    lse_block_stride,  # = block_size
    lse_token_stride,  # = 1
    src_lse_head_stride,  # = src_n_total_blocks * block_size
    dst_lse_head_stride,  # = sum(n_tokens_per_block)
    HEAD_DIM: tl.constexpr,  # head_dim
    MAX_N_BLOCKS_PER_OP: tl.constexpr,  # maximum number of src blocks per op
    TRITON_BLOCK_SIZE: tl.constexpr,  # block size for triton kernel
):
    # each program handles TRITON_BLOCK_SIZE tokens in a block
    # block parallelized over n_red_ops and n_head

    # total_programs = n_red_ops * n_head * n_programs_per_block
    program_id = tl.program_id(0)
    # print(f"program_id: {program_id}" + "=" * 20)
    n_programs_per_block = block_size // TRITON_BLOCK_SIZE

    block_offset = program_id % n_programs_per_block
    head_id = program_id // n_programs_per_block % n_head
    op_id = program_id // n_programs_per_block // n_head

    dst_token_offset = tl.load(dst_token_offsets + op_id)
    n_tokens_per_block = tl.load(n_tokens_per_block + op_id)
    n_tokens_for_current_prog = (
        n_tokens_per_block - block_offset * TRITON_BLOCK_SIZE
    )

    block_mask = tl.arange(0, MAX_N_BLOCKS_PER_OP) < tl.load(
        block_lengths + op_id
    )
    token_mask = tl.arange(0, TRITON_BLOCK_SIZE) < n_tokens_for_current_prog

    src_block_indices_in_buffer = tl.load(
        src_block_table
        + op_id * MAX_N_BLOCKS_PER_OP
        + tl.arange(0, MAX_N_BLOCKS_PER_OP),
        mask=block_mask,
    )  # [MAX_N_BLOCKS_PER_OP]
    # print("src_block_indices_in_buffer.shape", src_block_indices_in_buffer.shape)

    out_src_block_init_ptrs = (
        src_out_buffer
        + src_block_indices_in_buffer * out_block_stride
        + block_offset * TRITON_BLOCK_SIZE * out_token_stride
        + head_id * HEAD_DIM
    )  # [MAX_N_BLOCKS_PER_OP]
    # print("out_src_block_init_ptrs.shape", out_src_block_init_ptrs.shape)
    out_src_block_ptrs = (
        out_src_block_init_ptrs[:, None, None]
        + (tl.arange(0, TRITON_BLOCK_SIZE) * out_token_stride)[None, :, None]
        + tl.arange(0, HEAD_DIM)[None, None, :]
    )  # [MAX_N_BLOCKS_PER_OP, TRITON_BLOCK_SIZE, head_dim]
    # print("out_src_block_ptrs.shape", out_src_block_ptrs.shape)
    out_dst_block_init_ptr = (
        dst_out_buffer
        + (dst_token_offset + block_offset * TRITON_BLOCK_SIZE)
        * out_token_stride
        + head_id * HEAD_DIM
    )  # scalar
    out_dst_block_ptrs = (
        out_dst_block_init_ptr[None, :]
        + (tl.arange(0, TRITON_BLOCK_SIZE) * out_token_stride)[:, None]
        + tl.arange(0, HEAD_DIM)[None, :]
    )  # [TRITON_BLOCK_SIZE, head_dim]
    # print("out_dst_block_init_ptr[None, :].shape", out_dst_block_init_ptr[None, :].shape)
    # print("out_dst_block_ptrs.shape", out_dst_block_ptrs.shape)

    lse_src_block_init_ptrs = (
        src_lse_buffer
        + head_id * src_lse_head_stride
        + src_block_indices_in_buffer * lse_block_stride
        + block_offset * TRITON_BLOCK_SIZE * lse_token_stride
    )  # [MAX_N_BLOCKS_PER_OP]
    # print("lse_src_block_init_ptrs.shape", lse_src_block_init_ptrs.shape)
    lse_src_block_ptrs = (
        lse_src_block_init_ptrs[:, None]
        + tl.arange(0, TRITON_BLOCK_SIZE)[None, :]
    )  # [MAX_N_BLOCKS_PER_OP, TRITON_BLOCK_SIZE]
    # print("lse_src_block_ptrs.shape", lse_src_block_ptrs.shape)
    lse_dst_block_ptrs = (
        dst_lse_buffer
        + head_id * dst_lse_head_stride
        + (
            (dst_token_offset + block_offset * TRITON_BLOCK_SIZE)
            + tl.arange(0, TRITON_BLOCK_SIZE)
        )
        * lse_token_stride
    )  # [TRITON_BLOCK_SIZE]
    # print("lse_dst_block_ptr.shape", lse_dst_block_ptrs.shape)

    lse = tl.load(
        lse_src_block_ptrs,
        mask=block_mask[:, None] and token_mask[None, :],
        other=-float("inf"),
    )  # [MAX_N_BLOCKS_PER_OP, TRITON_BLOCK_SIZE]
    lse_finite_mask = lse != -float("inf") and lse != float("inf")
    lse_masked = tl.where(lse_finite_mask, lse, -float("inf"))
    lse_max = tl.max(lse_masked, axis=0)  # [TRITON_BLOCK_SIZE]
    new_lse = lse_max + tl.log(
        tl.sum(tl.exp(lse_masked - lse_max), axis=0)
    )  # [TRITON_BLOCK_SIZE]
    # reduce mask along the first dim
    lse_finite_mask_reduced = tl.reduce(
        lse_finite_mask, axis=0, combine_fn=_or_combine
    )
    new_lse = tl.where(lse_finite_mask_reduced, new_lse, -1)

    out = tl.load(
        out_src_block_ptrs,
        mask=(block_mask[:, None, None] and token_mask[None, :, None])
        and lse_finite_mask[:, :, None],
        other=0,
    )  # [MAX_N_BLOCKS_PER_OP, TRITON_BLOCK_SIZE, head_dim]
    out = tl.sum(
        out * tl.exp(lse_masked - new_lse)[:, :, None], axis=0
    )  # [TRITON_BLOCK_SIZE, head_dim]
    # store back
    tl.store(out_dst_block_ptrs, out, mask=token_mask[:, None])
    tl.store(lse_dst_block_ptrs, new_lse, mask=token_mask)


def flash_attn_update_out_lse(
    src_out_buffer,  # [src_n_total_blocks, block_size, n_head, head_dim]
    dst_out_buffer,  # [sum(n_tokens_per_block), n_head, head_dim]
    src_lse_buffer,  # [n_head, src_n_total_blocks, block_size]
    dst_lse_buffer,  # [n_head, sum(n_tokens_per_block)]
    n_head,  # number of attention heads
    src_block_table,  # (n_red_ops, MAX_N_BLOCKS_PER_OP)
    dst_token_offsets,  # (n_red_ops)
    n_tokens_per_block,  # (n_red_ops)
    block_lengths,  # length = n_red_ops, number of src blocks per op
    block_size,  # number of tokens in a block
    out_block_stride,  # = block_size * n_head * head_dim
    out_token_stride,  # = n_head * head_dim
    lse_block_stride,  # = block_size
    lse_token_stride,  # = 1
    src_lse_head_stride,  # = src_n_total_blocks * block_size
    dst_lse_head_stride,  # = sum(n_tokens_per_block)
    HEAD_DIM: int,  # head_dim
    max_n_blocks_per_op: int,  # maximum number of src blocks per op
    triton_block_size: int,  # block size for triton kernel
    require_cached: bool = False,
    return_key: bool = False,
):
    assert max_n_blocks_per_op <= 64, "Too many blocks for reduction"

    def _get_grid_dims():
        assert block_size % triton_block_size == 0
        n_programs_per_block = block_size // triton_block_size
        n_red_ops = len(block_lengths)
        return (n_red_ops * n_head * n_programs_per_block,)

    grid_dims = _get_grid_dims()
    if require_cached or return_key:
        if _flash_attn_update_out_lse.binder is None:
            target = driver.active.get_current_target()
            backend = make_backend(target)
            _flash_attn_update_out_lse.create_binder(backend)
        (
            bound_args,
            sig_and_spec,
            constexpr_vals,
            non_constexpr_vals,
            excess_kwargs,
        ) = _flash_attn_update_out_lse.binder(
            src_out_buffer,
            dst_out_buffer,
            src_lse_buffer,
            dst_lse_buffer,
            n_head,
            src_block_table,
            dst_token_offsets,
            n_tokens_per_block,
            block_lengths,
            block_size,
            out_block_stride,
            out_token_stride,
            lse_block_stride,
            lse_token_stride,
            src_lse_head_stride,
            dst_lse_head_stride,
            HEAD_DIM,
            max_n_blocks_per_op,
            triton_block_size,
            debug=False,
        )
        key = "".join(sig_and_spec) + str((constexpr_vals, excess_kwargs))
        device = driver.active.get_current_device()
        if require_cached:
            if key not in _flash_attn_update_out_lse.cache[device]:
                assert (
                    False
                ), "_flash_attn_update_out_lse key: {} not in cache, device: {}, existing keys: {}".format(
                    key,
                    device,
                    _flash_attn_update_out_lse.cache[device].keys(),
                )
    _flash_attn_update_out_lse[grid_dims](
        src_out_buffer,
        dst_out_buffer,
        src_lse_buffer,
        dst_lse_buffer,
        n_head,
        src_block_table,
        dst_token_offsets,
        n_tokens_per_block,
        block_lengths,
        block_size,
        out_block_stride,
        out_token_stride,
        lse_block_stride,
        lse_token_stride,
        src_lse_head_stride,
        dst_lse_head_stride,
        HEAD_DIM,
        max_n_blocks_per_op,
        triton_block_size,
    )
    if return_key:
        return key, device


@triton.jit
def _fused_copy_fp32(
    src_ptrs,
    src_offsets,
    src_strides,
    dst_ptrs,
    dst_offsets,
    dst_strides,
    BLOCK_SIZE: tl.constexpr,
    TRITON_BLOCK_SIZE: tl.constexpr,
):
    # total_programs = n_ptrs * (BLOCK_SIZE // TRITON_BLOCK_SIZE)
    prog_id = tl.program_id(0)
    n_prog_per_block = BLOCK_SIZE // TRITON_BLOCK_SIZE
    nth_ptr = prog_id // n_prog_per_block
    block_offset = prog_id % n_prog_per_block

    src_ptr = tl.load(src_ptrs + nth_ptr).to(tl.pointer_type(tl.float32))
    src_offset = tl.load(src_offsets + nth_ptr)
    src_stride = tl.load(src_strides + nth_ptr)

    dst_ptr = tl.load(dst_ptrs + nth_ptr).to(tl.pointer_type(tl.float32))
    dst_offset = tl.load(dst_offsets + nth_ptr)
    dst_stride = tl.load(dst_strides + nth_ptr)

    src_data = tl.load(
        src_ptr
        + src_offset * src_stride
        + block_offset * TRITON_BLOCK_SIZE
        + tl.arange(0, TRITON_BLOCK_SIZE)
    )
    tl.store(
        dst_ptr
        + dst_offset * dst_stride
        + block_offset * TRITON_BLOCK_SIZE
        + tl.arange(0, TRITON_BLOCK_SIZE),
        src_data,
    )


@triton.jit
def _fused_copy_fp32_varlen(
    src_ptrs,
    src_offsets,
    src_strides,
    dst_ptrs,
    dst_offsets,
    dst_strides,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    TRITON_BLOCK_SIZE: tl.constexpr,
):
    # total_programs = n_ptrs * (BLOCK_SIZE // TRITON_BLOCK_SIZE)
    prog_id = tl.program_id(0)
    n_prog_per_block = BLOCK_SIZE // TRITON_BLOCK_SIZE
    nth_ptr = prog_id // n_prog_per_block
    block_offset = prog_id % n_prog_per_block

    src_ptr = tl.load(src_ptrs + nth_ptr).to(tl.pointer_type(tl.float32))
    src_offset = tl.load(src_offsets + nth_ptr)
    src_stride = tl.load(src_strides + nth_ptr)

    dst_ptr = tl.load(dst_ptrs + nth_ptr).to(tl.pointer_type(tl.float32))
    dst_offset = tl.load(dst_offsets + nth_ptr)
    dst_stride = tl.load(dst_strides + nth_ptr)

    n_element_to_copy = tl.load(n_elements + nth_ptr)
    mask = (
        block_offset * TRITON_BLOCK_SIZE + tl.arange(0, TRITON_BLOCK_SIZE)
    ) < n_element_to_copy

    src_data = tl.load(
        src_ptr
        + src_offset * src_stride
        + block_offset * TRITON_BLOCK_SIZE
        + tl.arange(0, TRITON_BLOCK_SIZE),
        mask=mask,
    )
    tl.store(
        dst_ptr
        + dst_offset * dst_stride
        + block_offset * TRITON_BLOCK_SIZE
        + tl.arange(0, TRITON_BLOCK_SIZE),
        src_data,
        mask=mask,
    )


@triton.jit
def _fused_copy_fp32_varlen_dkv(
    src_ptrs,
    dst_ptrs,
    n_tokens,
    MAX_N_TOKENS: tl.constexpr,
    PER_TOKEN_SIZE: tl.constexpr,
    TRITON_BLOCK_SIZE: tl.constexpr,
):
    # total_programs = n_ptrs * (BLOCK_SIZE // TRITON_BLOCK_SIZE)
    prog_id = tl.program_id(0)
    n_prog_per_token = PER_TOKEN_SIZE // TRITON_BLOCK_SIZE
    n_prog_per_ptr = MAX_N_TOKENS * n_prog_per_token
    nth_ptr = prog_id // n_prog_per_ptr
    ptr_offset = prog_id % n_prog_per_ptr
    nth_token = ptr_offset // n_prog_per_token
    block_id = ptr_offset % n_prog_per_token

    n_tokens_to_copy = tl.load(n_tokens + nth_ptr)
    if nth_token < n_tokens_to_copy:
        src_ptr = tl.load(src_ptrs + nth_ptr).to(tl.pointer_type(tl.float32))
        dst_ptr = tl.load(dst_ptrs + nth_ptr).to(tl.pointer_type(tl.float32))
        offset = nth_token * PER_TOKEN_SIZE + block_id * TRITON_BLOCK_SIZE
        src_data = tl.load(
            src_ptr + offset + tl.arange(0, TRITON_BLOCK_SIZE),
        )
        tl.store(
            dst_ptr + offset + tl.arange(0, TRITON_BLOCK_SIZE),
            src_data,
        )


@triton.jit
def _fused_copy_fp32_varlen_lse(
    src_ptrs,
    src_token_offsets,
    src_head_strides,
    dst_ptrs,
    dst_token_offsets,
    dst_head_strides,
    n_tokens,
    HEAD_BLOCK_SIZE: tl.constexpr,
    MAX_N_TOKENS: tl.constexpr,
    TRITON_BLOCK_SIZE: tl.constexpr,
):
    # total_programs = n_ptrs * (MAX_N_TOKENS // TRITON_BLOCK_SIZE)
    prog_id = tl.program_id(0)
    n_prog_per_block = MAX_N_TOKENS // TRITON_BLOCK_SIZE
    nth_ptr = prog_id // n_prog_per_block
    block_offset = prog_id % n_prog_per_block

    src_ptr = tl.load(src_ptrs + nth_ptr).to(tl.pointer_type(tl.float32))
    src_token_offset = tl.load(src_token_offsets + nth_ptr)
    src_head_stride = tl.load(src_head_strides + nth_ptr)

    dst_ptr = tl.load(dst_ptrs + nth_ptr).to(tl.pointer_type(tl.float32))
    dst_token_offset = tl.load(dst_token_offsets + nth_ptr)
    dst_head_stride = tl.load(dst_head_strides + nth_ptr)

    n_token_to_copy = tl.load(n_tokens + nth_ptr)
    mask = (
        block_offset * TRITON_BLOCK_SIZE + tl.arange(0, TRITON_BLOCK_SIZE)
    ) < n_token_to_copy

    src_block_ptrs = (
        src_ptr
        + (tl.arange(0, HEAD_BLOCK_SIZE) * src_head_stride)[:, None]
        + (
            src_token_offset
            + block_offset * TRITON_BLOCK_SIZE
            + tl.arange(0, TRITON_BLOCK_SIZE)
        )[None, :]
    )
    dst_block_ptrs = (
        dst_ptr
        + (tl.arange(0, HEAD_BLOCK_SIZE) * dst_head_stride)[:, None]
        + (
            dst_token_offset
            + block_offset * TRITON_BLOCK_SIZE
            + tl.arange(0, TRITON_BLOCK_SIZE)
        )[None, :]
    )

    src_data = tl.load(
        src_block_ptrs,
        mask=mask[None, :],
    )
    tl.store(
        dst_block_ptrs,
        src_data,
        mask=mask[None, :],
    )


@triton.jit
def _fused_copy_fp16(
    src_ptrs,
    src_offsets,
    src_strides,
    dst_ptrs,
    dst_offsets,
    dst_strides,
    BLOCK_SIZE: tl.constexpr,
    TRITON_BLOCK_SIZE: tl.constexpr,
):
    # total_programs = n_ptrs * (BLOCK_SIZE // TRITON_BLOCK_SIZE)
    prog_id = tl.program_id(0)
    n_prog_per_block = BLOCK_SIZE // TRITON_BLOCK_SIZE
    nth_ptr = prog_id // n_prog_per_block
    block_offset = prog_id % n_prog_per_block

    src_ptr = tl.load(src_ptrs + nth_ptr).to(tl.pointer_type(tl.float16))
    src_offset = tl.load(src_offsets + nth_ptr)
    src_stride = tl.load(src_strides + nth_ptr)

    dst_ptr = tl.load(dst_ptrs + nth_ptr).to(tl.pointer_type(tl.float16))
    dst_offset = tl.load(dst_offsets + nth_ptr)
    dst_stride = tl.load(dst_strides + nth_ptr)

    src_data = tl.load(
        src_ptr
        + src_offset * src_stride
        + block_offset * TRITON_BLOCK_SIZE
        + tl.arange(0, TRITON_BLOCK_SIZE)
    )
    tl.store(
        dst_ptr
        + dst_offset * dst_stride
        + block_offset * TRITON_BLOCK_SIZE
        + tl.arange(0, TRITON_BLOCK_SIZE),
        src_data,
    )


@triton.jit
def _fused_copy_fp16_varlen(
    src_ptrs,
    src_offsets,
    src_strides,
    dst_ptrs,
    dst_offsets,
    dst_strides,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    TRITON_BLOCK_SIZE: tl.constexpr,
):
    # total_programs = n_ptrs * (BLOCK_SIZE // TRITON_BLOCK_SIZE)
    prog_id = tl.program_id(0)
    n_prog_per_block = BLOCK_SIZE // TRITON_BLOCK_SIZE
    nth_ptr = prog_id // n_prog_per_block
    block_offset = prog_id % n_prog_per_block

    src_ptr = tl.load(src_ptrs + nth_ptr).to(tl.pointer_type(tl.float16))
    src_offset = tl.load(src_offsets + nth_ptr)
    src_stride = tl.load(src_strides + nth_ptr)

    dst_ptr = tl.load(dst_ptrs + nth_ptr).to(tl.pointer_type(tl.float16))
    dst_offset = tl.load(dst_offsets + nth_ptr)
    dst_stride = tl.load(dst_strides + nth_ptr)

    n_element_to_copy = tl.load(n_elements + nth_ptr)
    mask = (
        block_offset * TRITON_BLOCK_SIZE + tl.arange(0, TRITON_BLOCK_SIZE)
    ) < n_element_to_copy

    src_data = tl.load(
        src_ptr
        + src_offset * src_stride
        + block_offset * TRITON_BLOCK_SIZE
        + tl.arange(0, TRITON_BLOCK_SIZE),
        mask=mask,
    )
    tl.store(
        dst_ptr
        + dst_offset * dst_stride
        + block_offset * TRITON_BLOCK_SIZE
        + tl.arange(0, TRITON_BLOCK_SIZE),
        src_data,
        mask=mask,
    )


@triton.jit
def _fused_copy_fp16_varlen_dkv(
    src_ptrs,
    dst_ptrs,
    n_tokens,
    MAX_N_TOKENS: tl.constexpr,
    PER_TOKEN_SIZE: tl.constexpr,
    TRITON_BLOCK_SIZE: tl.constexpr,
):
    # total_programs = n_ptrs * (BLOCK_SIZE // TRITON_BLOCK_SIZE)
    prog_id = tl.program_id(0)
    n_prog_per_token = PER_TOKEN_SIZE // TRITON_BLOCK_SIZE
    n_prog_per_ptr = MAX_N_TOKENS * n_prog_per_token
    nth_ptr = prog_id // n_prog_per_ptr
    ptr_offset = prog_id % n_prog_per_ptr
    nth_token = ptr_offset // n_prog_per_token
    block_id = ptr_offset % n_prog_per_token

    n_tokens_to_copy = tl.load(n_tokens + nth_ptr)
    if nth_token < n_tokens_to_copy:
        src_ptr = tl.load(src_ptrs + nth_ptr).to(tl.pointer_type(tl.float16))
        dst_ptr = tl.load(dst_ptrs + nth_ptr).to(tl.pointer_type(tl.float16))
        offset = nth_token * PER_TOKEN_SIZE + block_id * TRITON_BLOCK_SIZE
        src_data = tl.load(
            src_ptr + offset + tl.arange(0, TRITON_BLOCK_SIZE),
        )
        tl.store(
            dst_ptr + offset + tl.arange(0, TRITON_BLOCK_SIZE),
            src_data,
        )


@triton.jit
def _fused_copy_fp16_varlen_lse(
    src_ptrs,
    src_token_offsets,
    src_head_strides,
    dst_ptrs,
    dst_token_offsets,
    dst_head_strides,
    n_tokens,
    HEAD_BLOCK_SIZE: tl.constexpr,
    MAX_N_TOKENS: tl.constexpr,
    TRITON_BLOCK_SIZE: tl.constexpr,
):
    # total_programs = n_ptrs * (MAX_N_TOKENS // TRITON_BLOCK_SIZE)
    prog_id = tl.program_id(0)
    n_prog_per_block = MAX_N_TOKENS // TRITON_BLOCK_SIZE
    nth_ptr = prog_id // n_prog_per_block
    block_offset = prog_id % n_prog_per_block

    src_ptr = tl.load(src_ptrs + nth_ptr).to(tl.pointer_type(tl.float16))
    src_token_offset = tl.load(src_token_offsets + nth_ptr)
    src_head_stride = tl.load(src_head_strides + nth_ptr)

    dst_ptr = tl.load(dst_ptrs + nth_ptr).to(tl.pointer_type(tl.float16))
    dst_token_offset = tl.load(dst_token_offsets + nth_ptr)
    dst_head_stride = tl.load(dst_head_strides + nth_ptr)

    n_token_to_copy = tl.load(n_tokens + nth_ptr)
    mask = (
        block_offset * TRITON_BLOCK_SIZE + tl.arange(0, TRITON_BLOCK_SIZE)
    ) < n_token_to_copy

    src_block_ptrs = (
        src_ptr
        + (tl.arange(0, HEAD_BLOCK_SIZE) * src_head_stride)[:, None]
        + (
            src_token_offset
            + block_offset * TRITON_BLOCK_SIZE
            + tl.arange(0, TRITON_BLOCK_SIZE)
        )[None, :]
    )
    dst_block_ptrs = (
        dst_ptr
        + (tl.arange(0, HEAD_BLOCK_SIZE) * dst_head_stride)[:, None]
        + (
            dst_token_offset
            + block_offset * TRITON_BLOCK_SIZE
            + tl.arange(0, TRITON_BLOCK_SIZE)
        )[None, :]
    )

    src_data = tl.load(
        src_block_ptrs,
        mask=mask[None, :],
    )
    tl.store(
        dst_block_ptrs,
        src_data,
        mask=mask[None, :],
    )


@triton.jit
def _fused_copy_bf16(
    src_ptrs,
    src_offsets,
    src_strides,
    dst_ptrs,
    dst_offsets,
    dst_strides,
    BLOCK_SIZE: tl.constexpr,
    TRITON_BLOCK_SIZE: tl.constexpr,
):
    # total_programs = n_ptrs * (BLOCK_SIZE // TRITON_BLOCK_SIZE)
    prog_id = tl.program_id(0)
    n_prog_per_block = BLOCK_SIZE // TRITON_BLOCK_SIZE
    nth_ptr = prog_id // n_prog_per_block
    block_offset = prog_id % n_prog_per_block

    src_ptr = tl.load(src_ptrs + nth_ptr).to(tl.pointer_type(tl.bfloat16))
    src_offset = tl.load(src_offsets + nth_ptr)
    src_stride = tl.load(src_strides + nth_ptr)

    dst_ptr = tl.load(dst_ptrs + nth_ptr).to(tl.pointer_type(tl.bfloat16))
    dst_offset = tl.load(dst_offsets + nth_ptr)
    dst_stride = tl.load(dst_strides + nth_ptr)

    src_data = tl.load(
        src_ptr
        + src_offset * src_stride
        + block_offset * TRITON_BLOCK_SIZE
        + tl.arange(0, TRITON_BLOCK_SIZE)
    )
    tl.store(
        dst_ptr
        + dst_offset * dst_stride
        + block_offset * TRITON_BLOCK_SIZE
        + tl.arange(0, TRITON_BLOCK_SIZE),
        src_data,
    )


@triton.jit
def _fused_copy_bf16_varlen(
    src_ptrs,
    src_offsets,
    src_strides,
    dst_ptrs,
    dst_offsets,
    dst_strides,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    TRITON_BLOCK_SIZE: tl.constexpr,
):
    # total_programs = n_ptrs * (BLOCK_SIZE // TRITON_BLOCK_SIZE)
    prog_id = tl.program_id(0)
    n_prog_per_block = BLOCK_SIZE // TRITON_BLOCK_SIZE
    nth_ptr = prog_id // n_prog_per_block
    block_offset = prog_id % n_prog_per_block

    src_ptr = tl.load(src_ptrs + nth_ptr).to(tl.pointer_type(tl.bfloat16))
    src_offset = tl.load(src_offsets + nth_ptr)
    src_stride = tl.load(src_strides + nth_ptr)

    dst_ptr = tl.load(dst_ptrs + nth_ptr).to(tl.pointer_type(tl.bfloat16))
    dst_offset = tl.load(dst_offsets + nth_ptr)
    dst_stride = tl.load(dst_strides + nth_ptr)

    n_element_to_copy = tl.load(n_elements + nth_ptr)
    mask = (
        block_offset * TRITON_BLOCK_SIZE + tl.arange(0, TRITON_BLOCK_SIZE)
    ) < n_element_to_copy

    src_data = tl.load(
        src_ptr
        + src_offset * src_stride
        + block_offset * TRITON_BLOCK_SIZE
        + tl.arange(0, TRITON_BLOCK_SIZE),
        mask=mask,
    )
    tl.store(
        dst_ptr
        + dst_offset * dst_stride
        + block_offset * TRITON_BLOCK_SIZE
        + tl.arange(0, TRITON_BLOCK_SIZE),
        src_data,
        mask=mask,
    )


@triton.jit
def _fused_copy_bf16_varlen_dkv(
    src_ptrs,
    dst_ptrs,
    n_tokens,
    MAX_N_TOKENS: tl.constexpr,
    PER_TOKEN_SIZE: tl.constexpr,
    TRITON_BLOCK_SIZE: tl.constexpr,
):
    # total_programs = n_ptrs * (BLOCK_SIZE // TRITON_BLOCK_SIZE)
    prog_id = tl.program_id(0)
    n_prog_per_token = PER_TOKEN_SIZE // TRITON_BLOCK_SIZE
    n_prog_per_ptr = MAX_N_TOKENS * n_prog_per_token
    nth_ptr = prog_id // n_prog_per_ptr
    ptr_offset = prog_id % n_prog_per_ptr
    nth_token = ptr_offset // n_prog_per_token
    block_id = ptr_offset % n_prog_per_token

    n_tokens_to_copy = tl.load(n_tokens + nth_ptr)
    if nth_token < n_tokens_to_copy:
        src_ptr = tl.load(src_ptrs + nth_ptr).to(tl.pointer_type(tl.bfloat16))
        dst_ptr = tl.load(dst_ptrs + nth_ptr).to(tl.pointer_type(tl.bfloat16))
        offset = nth_token * PER_TOKEN_SIZE + block_id * TRITON_BLOCK_SIZE
        src_data = tl.load(
            src_ptr + offset + tl.arange(0, TRITON_BLOCK_SIZE),
        )
        tl.store(
            dst_ptr + offset + tl.arange(0, TRITON_BLOCK_SIZE),
            src_data,
        )


@triton.jit
def _fused_copy_bf16_varlen_lse(
    src_ptrs,
    src_token_offsets,
    src_head_strides,
    dst_ptrs,
    dst_token_offsets,
    dst_head_strides,
    n_tokens,
    HEAD_BLOCK_SIZE: tl.constexpr,
    MAX_N_TOKENS: tl.constexpr,
    TRITON_BLOCK_SIZE: tl.constexpr,
):
    # total_programs = n_ptrs * (MAX_N_TOKENS // TRITON_BLOCK_SIZE)
    prog_id = tl.program_id(0)
    n_prog_per_block = MAX_N_TOKENS // TRITON_BLOCK_SIZE
    nth_ptr = prog_id // n_prog_per_block
    block_offset = prog_id % n_prog_per_block

    src_ptr = tl.load(src_ptrs + nth_ptr).to(tl.pointer_type(tl.bfloat16))
    src_token_offset = tl.load(src_token_offsets + nth_ptr)
    src_head_stride = tl.load(src_head_strides + nth_ptr)

    dst_ptr = tl.load(dst_ptrs + nth_ptr).to(tl.pointer_type(tl.bfloat16))
    dst_token_offset = tl.load(dst_token_offsets + nth_ptr)
    dst_head_stride = tl.load(dst_head_strides + nth_ptr)

    n_token_to_copy = tl.load(n_tokens + nth_ptr)
    mask = (
        block_offset * TRITON_BLOCK_SIZE + tl.arange(0, TRITON_BLOCK_SIZE)
    ) < n_token_to_copy

    src_block_ptrs = (
        src_ptr
        + (tl.arange(0, HEAD_BLOCK_SIZE) * src_head_stride)[:, None]
        + (
            src_token_offset
            + block_offset * TRITON_BLOCK_SIZE
            + tl.arange(0, TRITON_BLOCK_SIZE)
        )[None, :]
    )
    dst_block_ptrs = (
        dst_ptr
        + (tl.arange(0, HEAD_BLOCK_SIZE) * dst_head_stride)[:, None]
        + (
            dst_token_offset
            + block_offset * TRITON_BLOCK_SIZE
            + tl.arange(0, TRITON_BLOCK_SIZE)
        )[None, :]
    )

    src_data = tl.load(
        src_block_ptrs,
        mask=mask[None, :],
    )
    tl.store(
        dst_block_ptrs,
        src_data,
        mask=mask[None, :],
    )


def fused_copy_bf16(
    src_ptrs,
    src_offsets,
    src_strides,
    dst_ptrs,
    dst_offsets,
    dst_strides,
    block_size: int,
    triton_block_size: int,
):
    n_ptrs = len(src_ptrs)
    _fused_copy_bf16[(n_ptrs * (block_size // triton_block_size),)](
        src_ptrs,
        src_offsets,
        src_strides,
        dst_ptrs,
        dst_offsets,
        dst_strides,
        block_size,
        triton_block_size,
    )


def fused_copy_fp16(
    src_ptrs,
    src_offsets,
    src_strides,
    dst_ptrs,
    dst_offsets,
    dst_strides,
    block_size: int,
    triton_block_size: int,
):
    n_ptrs = len(src_ptrs)
    _fused_copy_fp16[(n_ptrs * (block_size // triton_block_size),)](
        src_ptrs,
        src_offsets,
        src_strides,
        dst_ptrs,
        dst_offsets,
        dst_strides,
        block_size,
        triton_block_size,
    )


def fused_copy_bf16_varlen(
    src_ptrs,
    src_offsets,
    src_strides,
    dst_ptrs,
    dst_offsets,
    dst_strides,
    n_elements,
    block_size: int,
    triton_block_size: int,
):
    n_ptrs = len(src_ptrs)
    _fused_copy_bf16_varlen[(n_ptrs * (block_size // triton_block_size),)](
        src_ptrs,
        src_offsets,
        src_strides,
        dst_ptrs,
        dst_offsets,
        dst_strides,
        n_elements,
        block_size,
        triton_block_size,
    )


def fused_copy_bf16_varlen_dkv(
    src_ptrs,
    dst_ptrs,
    n_tokens,
    max_n_tokens: int,
    per_token_size: int,
    triton_block_size: int,
):
    n_ptrs = len(src_ptrs)
    assert per_token_size % triton_block_size == 0
    n_progs = n_ptrs * max_n_tokens * (per_token_size // triton_block_size)
    _fused_copy_bf16_varlen_dkv[(n_progs,)](
        src_ptrs,
        dst_ptrs,
        n_tokens,
        max_n_tokens,
        per_token_size,
        triton_block_size,
    )


def fused_copy_bf16_varlen_lse(
    src_ptrs,
    src_token_offsets,
    src_head_strides,
    dst_ptrs,
    dst_token_offsets,
    dst_head_strides,
    n_tokens,
    head_block_size: int,
    block_size: int,
    triton_block_size: int,
):
    n_ptrs = len(src_ptrs)
    assert block_size % triton_block_size == 0
    _fused_copy_bf16_varlen_lse[(n_ptrs * (block_size // triton_block_size),)](
        src_ptrs,
        src_token_offsets,
        src_head_strides,
        dst_ptrs,
        dst_token_offsets,
        dst_head_strides,
        n_tokens,
        head_block_size,
        block_size,
        triton_block_size,
    )


def fused_copy_fp16_varlen(
    src_ptrs,
    src_offsets,
    src_strides,
    dst_ptrs,
    dst_offsets,
    dst_strides,
    n_elements,
    block_size: int,
    triton_block_size: int,
):
    n_ptrs = len(src_ptrs)
    _fused_copy_fp16_varlen[(n_ptrs * (block_size // triton_block_size),)](
        src_ptrs,
        src_offsets,
        src_strides,
        dst_ptrs,
        dst_offsets,
        dst_strides,
        n_elements,
        block_size,
        triton_block_size,
    )


def fused_copy_fp16_varlen_dkv(
    src_ptrs,
    dst_ptrs,
    n_tokens,
    max_n_tokens: int,
    per_token_size: int,
    triton_block_size: int,
):
    n_ptrs = len(src_ptrs)
    assert per_token_size % triton_block_size == 0
    n_progs = n_ptrs * max_n_tokens * (per_token_size // triton_block_size)
    _fused_copy_fp16_varlen_dkv[(n_progs,)](
        src_ptrs,
        dst_ptrs,
        n_tokens,
        max_n_tokens,
        per_token_size,
        triton_block_size,
    )


def fused_copy_fp16_varlen_lse(
    src_ptrs,
    src_token_offsets,
    src_head_strides,
    dst_ptrs,
    dst_token_offsets,
    dst_head_strides,
    n_tokens,
    head_block_size: int,
    block_size: int,
    triton_block_size: int,
):
    n_ptrs = len(src_ptrs)
    assert block_size % triton_block_size == 0
    _fused_copy_fp16_varlen_lse[(n_ptrs * (block_size // triton_block_size),)](
        src_ptrs,
        src_token_offsets,
        src_head_strides,
        dst_ptrs,
        dst_token_offsets,
        dst_head_strides,
        n_tokens,
        head_block_size,
        block_size,
        triton_block_size,
    )


def fused_copy_fp32(
    src_ptrs,
    src_offsets,
    src_strides,
    dst_ptrs,
    dst_offsets,
    dst_strides,
    block_size: int,
    triton_block_size: int,
):
    n_ptrs = len(src_ptrs)
    _fused_copy_fp32[(n_ptrs * (block_size // triton_block_size),)](
        src_ptrs,
        src_offsets,
        src_strides,
        dst_ptrs,
        dst_offsets,
        dst_strides,
        block_size,
        triton_block_size,
    )


def fused_copy_fp32_varlen(
    src_ptrs,
    src_offsets,
    src_strides,
    dst_ptrs,
    dst_offsets,
    dst_strides,
    n_elements,
    block_size: int,
    triton_block_size: int,
):
    n_ptrs = len(src_ptrs)
    _fused_copy_fp32_varlen[(n_ptrs * (block_size // triton_block_size),)](
        src_ptrs,
        src_offsets,
        src_strides,
        dst_ptrs,
        dst_offsets,
        dst_strides,
        n_elements,
        block_size,
        triton_block_size,
    )


def fused_copy_fp32_varlen_dkv(
    src_ptrs,
    dst_ptrs,
    n_tokens,
    max_n_tokens: int,
    per_token_size: int,
    triton_block_size: int,
):
    n_ptrs = len(src_ptrs)
    assert per_token_size % triton_block_size == 0
    n_progs = n_ptrs * max_n_tokens * (per_token_size // triton_block_size)
    _fused_copy_fp32_varlen_dkv[(n_progs,)](
        src_ptrs,
        dst_ptrs,
        n_tokens,
        max_n_tokens,
        per_token_size,
        triton_block_size,
    )


def fused_copy_fp32_varlen_lse(
    src_ptrs,
    src_token_offsets,
    src_head_strides,
    dst_ptrs,
    dst_token_offsets,
    dst_head_strides,
    n_tokens,
    head_block_size: int,
    block_size: int,
    triton_block_size: int,
):
    n_ptrs = len(src_ptrs)
    assert block_size % triton_block_size == 0
    _fused_copy_fp32_varlen_lse[(n_ptrs * (block_size // triton_block_size),)](
        src_ptrs,
        src_token_offsets,
        src_head_strides,
        dst_ptrs,
        dst_token_offsets,
        dst_head_strides,
        n_tokens,
        head_block_size,
        block_size,
        triton_block_size,
    )


@triton.jit
def _fused_blockwise_sum_bf16(
    src_ptrs,  # [n_blocks]
    src_offsets,  # [n_blocks, MAX_N_OPS_PER_BLOCK]
    src_strides,  # [n_blocks]
    n_ops_per_block,  # [n_blocks]
    dst_ptrs,  # [n_blocks]
    dst_offsets,  # [n_blocks]
    dst_strides,  # [n_blocks]
    n_elements,  # [n_blocks]
    MAX_N_OPS_PER_BLOCK: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    TRITON_BLOCK_SIZE: tl.constexpr,
):
    # total_programs = n_ptrs * (BLOCK_SIZE // TRITON_BLOCK_SIZE)
    prog_id = tl.program_id(0)
    n_prog_per_block = BLOCK_SIZE // TRITON_BLOCK_SIZE
    nth_ptr = prog_id // n_prog_per_block
    block_offset = prog_id % n_prog_per_block

    curr_src_offsets = tl.load(
        src_offsets
        + nth_ptr * MAX_N_OPS_PER_BLOCK
        + tl.arange(0, MAX_N_OPS_PER_BLOCK)
    )

    n_element = tl.load(n_elements + nth_ptr)

    token_mask = (
        block_offset * TRITON_BLOCK_SIZE + tl.arange(0, TRITON_BLOCK_SIZE)
    ) < n_element

    n_ops = tl.load(n_ops_per_block + nth_ptr)
    ops_mask = tl.arange(0, MAX_N_OPS_PER_BLOCK) < n_ops

    src_buffer_ptr = tl.load(src_ptrs + nth_ptr).to(
        tl.pointer_type(tl.bfloat16)
    )
    src_stride = tl.load(src_strides + nth_ptr)

    src_data = tl.load(
        src_buffer_ptr
        + curr_src_offsets[:, None] * src_stride  # [MAX_N_OPS_PER_BLOCK, 1]
        + (block_offset * TRITON_BLOCK_SIZE + tl.arange(0, TRITON_BLOCK_SIZE))[
            None, :
        ],  # [1, TRITON_BLOCK_SIZE]
        mask=token_mask[None, :]
        & ops_mask[:, None],  # [MAX_N_OPS_PER_BLOCK, TRITON_BLOCK_SIZE]
        other=0,
    )

    sum_src_data = tl.sum(src_data, axis=0)  # [TRITON_BLOCK_SIZE]

    dst_ptr = tl.load(dst_ptrs + nth_ptr).to(tl.pointer_type(tl.bfloat16))
    dst_offset = tl.load(dst_offsets + nth_ptr)
    dst_stride = tl.load(dst_strides + nth_ptr)

    tl.store(
        dst_ptr
        + dst_offset * dst_stride
        + block_offset * TRITON_BLOCK_SIZE
        + tl.arange(0, TRITON_BLOCK_SIZE),
        sum_src_data,
        mask=token_mask,
    )


@triton.jit
def _fused_blockwise_sum_fp16(
    src_ptrs,  # [n_blocks]
    src_offsets,  # [n_blocks, MAX_N_OPS_PER_BLOCK]
    src_strides,  # [n_blocks]
    n_ops_per_block,  # [n_blocks]
    dst_ptrs,  # [n_blocks]
    dst_offsets,  # [n_blocks]
    dst_strides,  # [n_blocks]
    n_elements,  # [n_blocks]
    MAX_N_OPS_PER_BLOCK: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    TRITON_BLOCK_SIZE: tl.constexpr,
):
    # total_programs = n_ptrs * (BLOCK_SIZE // TRITON_BLOCK_SIZE)
    prog_id = tl.program_id(0)
    n_prog_per_block = BLOCK_SIZE // TRITON_BLOCK_SIZE
    nth_ptr = prog_id // n_prog_per_block
    block_offset = prog_id % n_prog_per_block

    curr_src_offsets = tl.load(
        src_offsets
        + nth_ptr * MAX_N_OPS_PER_BLOCK
        + tl.arange(0, MAX_N_OPS_PER_BLOCK)
    )

    n_element = tl.load(n_elements + nth_ptr)

    token_mask = (
        block_offset * TRITON_BLOCK_SIZE + tl.arange(0, TRITON_BLOCK_SIZE)
    ) < n_element

    n_ops = tl.load(n_ops_per_block + nth_ptr)
    ops_mask = tl.arange(0, MAX_N_OPS_PER_BLOCK) < n_ops

    src_buffer_ptr = tl.load(src_ptrs + nth_ptr).to(
        tl.pointer_type(tl.float16)
    )
    src_stride = tl.load(src_strides + nth_ptr)

    src_data = tl.load(
        src_buffer_ptr
        + curr_src_offsets[:, None] * src_stride  # [MAX_N_OPS_PER_BLOCK, 1]
        + (block_offset * TRITON_BLOCK_SIZE + tl.arange(0, TRITON_BLOCK_SIZE))[
            None, :
        ],  # [1, TRITON_BLOCK_SIZE]
        mask=token_mask[None, :]
        and ops_mask[:, None],  # [MAX_N_OPS_PER_BLOCK, TRITON_BLOCK_SIZE]
        other=0,
    )

    sum_src_data = tl.sum(src_data, axis=0)  # [TRITON_BLOCK_SIZE]

    dst_ptr = tl.load(dst_ptrs + nth_ptr).to(tl.pointer_type(tl.float16))
    dst_offset = tl.load(dst_offsets + nth_ptr)
    dst_stride = tl.load(dst_strides + nth_ptr)

    tl.store(
        dst_ptr
        + dst_offset * dst_stride
        + block_offset * TRITON_BLOCK_SIZE
        + tl.arange(0, TRITON_BLOCK_SIZE),
        sum_src_data,
        mask=token_mask,
    )


@triton.jit
def _fused_blockwise_sum_fp32(
    src_ptrs,  # [n_blocks]
    src_offsets,  # [n_blocks, MAX_N_OPS_PER_BLOCK]
    src_strides,  # [n_blocks]
    n_ops_per_block,  # [n_blocks]
    dst_ptrs,  # [n_blocks]
    dst_offsets,  # [n_blocks]
    dst_strides,  # [n_blocks]
    n_elements,  # [n_blocks]
    MAX_N_OPS_PER_BLOCK: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    TRITON_BLOCK_SIZE: tl.constexpr,
):
    # total_programs = n_ptrs * (BLOCK_SIZE // TRITON_BLOCK_SIZE)
    prog_id = tl.program_id(0)
    n_prog_per_block = BLOCK_SIZE // TRITON_BLOCK_SIZE
    nth_ptr = prog_id // n_prog_per_block
    block_offset = prog_id % n_prog_per_block

    curr_src_offsets = tl.load(
        src_offsets
        + nth_ptr * MAX_N_OPS_PER_BLOCK
        + tl.arange(0, MAX_N_OPS_PER_BLOCK)
    )

    n_element = tl.load(n_elements + nth_ptr)

    token_mask = (
        block_offset * TRITON_BLOCK_SIZE + tl.arange(0, TRITON_BLOCK_SIZE)
    ) < n_element

    n_ops = tl.load(n_ops_per_block + nth_ptr)
    ops_mask = tl.arange(0, MAX_N_OPS_PER_BLOCK) < n_ops

    src_buffer_ptr = tl.load(src_ptrs + nth_ptr).to(
        tl.pointer_type(tl.float32)
    )
    src_stride = tl.load(src_strides + nth_ptr)

    src_data = tl.load(
        src_buffer_ptr
        + curr_src_offsets[:, None] * src_stride  # [MAX_N_OPS_PER_BLOCK, 1]
        + (block_offset * TRITON_BLOCK_SIZE + tl.arange(0, TRITON_BLOCK_SIZE))[
            None, :
        ],  # [1, TRITON_BLOCK_SIZE]
        mask=token_mask[None, :]
        and ops_mask[:, None],  # [MAX_N_OPS_PER_BLOCK, TRITON_BLOCK_SIZE]
        other=0,
    )

    sum_src_data = tl.sum(src_data, axis=0)  # [TRITON_BLOCK_SIZE]

    dst_ptr = tl.load(dst_ptrs + nth_ptr).to(tl.pointer_type(tl.float32))
    dst_offset = tl.load(dst_offsets + nth_ptr)
    dst_stride = tl.load(dst_strides + nth_ptr)

    tl.store(
        dst_ptr
        + dst_offset * dst_stride
        + block_offset * TRITON_BLOCK_SIZE
        + tl.arange(0, TRITON_BLOCK_SIZE),
        sum_src_data,
        mask=token_mask,
    )


def fused_blockwise_sum(
    src_ptrs,
    src_offsets,
    src_strides,
    n_ops_per_block,
    dst_ptrs,
    dst_offsets,
    dst_strides,
    n_elements,
    dtype,
    max_n_ops_per_block: int,
    block_size: int,
    triton_block_size: int,
    require_cached: bool = False,
):
    n_ptrs = len(src_ptrs)
    assert block_size % triton_block_size == 0
    if not isinstance(dtype, str):
        dtype = str(dtype)
    if dtype == "torch.bfloat16":
        fn = _fused_blockwise_sum_bf16
    elif dtype == "torch.float16":
        fn = _fused_blockwise_sum_fp16
    elif dtype == "torch.float32":
        fn = _fused_blockwise_sum_fp32
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")
    if require_cached:
        if fn.binder is None:
            target = driver.active.get_current_target()
            backend = make_backend(target)
            fn.create_binder(backend)
        (
            bound_args,
            sig_and_spec,
            constexpr_vals,
            non_constexpr_vals,
            excess_kwargs,
        ) = fn.binder(
            src_ptrs,
            src_offsets,
            src_strides,
            n_ops_per_block,
            dst_ptrs,
            dst_offsets,
            dst_strides,
            n_elements,
            max_n_ops_per_block,
            block_size,
            triton_block_size,
            debug=False,
        )
        key = "".join(sig_and_spec) + str((constexpr_vals, excess_kwargs))
        device = driver.active.get_current_device()
        if key not in fn.cache[device]:
            assert (
                False
            ), "FusedBlockwiseSum: key: {} not in cache, device: {}, existing keys: {}".format(
                key, device, fn.cache[device].keys()
            )
    fn[(n_ptrs * (block_size // triton_block_size),)](
        src_ptrs,
        src_offsets,
        src_strides,
        n_ops_per_block,
        dst_ptrs,
        dst_offsets,
        dst_strides,
        n_elements,
        max_n_ops_per_block,
        block_size,
        triton_block_size,
    )


def _gen_flash_attn_update_out_lse_inputs(
    n_head: int,
    head_dim: int,
    max_n_blocks_per_op: int,
    block_size: int,
    triton_block_size: int,
    dtype: torch.dtype,
):
    n_src_blocks_per_op = [max_n_blocks_per_op] * 2
    src_buffer_size_n_blocks = 256 * 2
    n_tokens_per_block = torch.randint(
        1,
        block_size,
        (len(n_src_blocks_per_op),),
        dtype=torch.int32,
        device="cuda",
    )
    dst_token_offsets = (
        F.pad(n_tokens_per_block, (1, 0), value=0)
        .cumsum(0)[:-1]
        .to(torch.int32)
    )

    src_out_buffer = torch.randn(
        src_buffer_size_n_blocks,
        block_size,
        n_head,
        head_dim,
        dtype=dtype,
        device="cuda",
    )
    dst_out_buffer = torch.zeros(
        torch.sum(n_tokens_per_block).item(),
        n_head,
        head_dim,
        dtype=dtype,
        device="cuda",
    )
    src_lse_buffer = torch.randn(
        n_head,
        src_buffer_size_n_blocks,
        block_size,
        dtype=torch.float32,
        device="cuda",
    )
    dst_lse_buffer = torch.zeros(
        n_head,
        torch.sum(n_tokens_per_block).item(),
        dtype=torch.float32,
        device="cuda",
    )

    src_block_table_combined = torch.tensor(
        np.random.choice(
            src_buffer_size_n_blocks,
            ((sum(n_src_blocks_per_op),)),
            replace=False,
        ),
        dtype=torch.int32,
    ).tolist()

    src_block_table = []
    c = 0
    for n in n_src_blocks_per_op:
        curr_op_src_blocks = src_block_table_combined[c : c + n] + [0] * (
            max_n_blocks_per_op - n
        )
        src_block_table.append(curr_op_src_blocks)
        c += n
    src_block_table = torch.tensor(
        src_block_table, dtype=torch.int32, device="cuda"
    )

    block_lengths_tensor = torch.tensor(
        n_src_blocks_per_op, dtype=torch.int32, device="cuda"
    )
    return (
        src_out_buffer,
        dst_out_buffer,
        src_lse_buffer,
        dst_lse_buffer,
        n_head,
        src_block_table,
        dst_token_offsets,
        n_tokens_per_block,
        block_lengths_tensor,
        block_size,
        block_size * n_head * head_dim,
        n_head * head_dim,
        block_size,
        1,
        src_buffer_size_n_blocks * block_size,
        torch.sum(n_tokens_per_block).item(),
        head_dim,
        max_n_blocks_per_op,
        triton_block_size,
    )


def _gen_memcpy_inputs(
    block_size: int, triton_block_size: int, n_blocks: int, dtype: torch.dtype
):
    buffers = []

    n_blocks = 8

    buffer = torch.randn(n_blocks * 2, block_size, dtype=dtype, device="cuda")
    buffers.append(buffer)
    dst_buffer = torch.randn_like(buffer, dtype=dtype)
    buffers.append(dst_buffer)
    buffer_ptrs = torch.tensor(
        [buffer.data_ptr() for _ in range(n_blocks)],
        device="cuda",
    )
    dst_buffer_ptrs = torch.tensor(
        [dst_buffer.data_ptr() for _ in range(n_blocks)],
        device="cuda",
    )
    src_ptrs = buffer_ptrs
    dst_ptrs = dst_buffer_ptrs
    block_offsets = torch.tensor(
        np.random.choice(n_blocks * 2, n_blocks, replace=False),
        dtype=torch.int32,
        device="cuda",
    )
    dst_block_offsets = torch.tensor(
        np.random.choice(n_blocks * 2, n_blocks, replace=False),
        dtype=torch.int32,
        device="cuda",
    )
    block_strides = torch.tensor(
        [block_size for _ in range(n_blocks)],
        dtype=torch.int32,
        device="cuda",
    )
    dst_block_strides = torch.tensor(
        [block_size for _ in range(n_blocks)],
        dtype=torch.int32,
        device="cuda",
    )
    src_offsets = block_offsets
    dst_offsets = dst_block_offsets
    src_strides = block_strides
    dst_strides = dst_block_strides
    n_elements = torch.randint(
        1, block_size + 1, (n_blocks,), dtype=torch.int32, device="cuda"
    )
    return (
        buffers,
        src_ptrs,
        src_offsets,
        src_strides,
        dst_ptrs,
        dst_offsets,
        dst_strides,
        n_elements,
        block_size,
        triton_block_size,
    )


def _gen_memcpy_lse_inputs(
    block_size: int, triton_block_size: int, n_blocks: int, dtype: torch.dtype
):
    head_block_size = 1
    n_tokens = n_blocks * block_size
    src_buffer = torch.randn(
        head_block_size, n_tokens, dtype=dtype, device="cuda"
    )
    dst_buffer = torch.randn_like(src_buffer)
    src_offsets = torch.arange(0, n_blocks, device="cuda") * (
        n_tokens // n_blocks
    )
    n_tokens_tensor = torch.randint(
        1, block_size + 1, (n_blocks,), dtype=torch.int32, device="cuda"
    )
    dst_offsets = torch.arange(0, n_blocks, device="cuda") * (
        n_tokens // n_blocks
    )

    src_ptrs = torch.tensor(
        [src_buffer.data_ptr() for _ in range(n_blocks)], device="cuda"
    )
    dst_ptrs = torch.tensor(
        [dst_buffer.data_ptr() for _ in range(n_blocks)], device="cuda"
    )
    head_strides = torch.tensor(
        [n_tokens for _ in range(n_blocks)], dtype=torch.int32, device="cuda"
    )
    return (
        [src_buffer, dst_buffer],
        src_ptrs,
        src_offsets,
        head_strides,
        dst_ptrs,
        dst_offsets,
        head_strides,
        n_tokens_tensor,
        head_block_size,
        block_size,
        triton_block_size,
    )


def _gen_memcpy_dkv_inputs(
    block_size: int, kv_head_block_size: int, head_dim: int, dtype: torch.dtype
):
    max_n_blocks = 8
    src_block_indices = [3, 6, 2, 0]
    dst_block_indices = [7, 1, 5, 4]

    src_buffer = torch.randn(
        max_n_blocks,
        block_size,
        2,
        kv_head_block_size,
        head_dim,
        dtype=dtype,
        device="cuda",
    )
    dst_buffer = torch.zeros_like(src_buffer)

    src_ptrs = [src_buffer[idx].data_ptr() for idx in src_block_indices]
    dst_ptrs = [dst_buffer[idx].data_ptr() for idx in dst_block_indices]
    n_tokens = torch.randint(
        1,
        block_size + 1,
        (len(src_block_indices),),
        dtype=torch.int32,
        device="cuda",
    )
    per_token_size = 2 * kv_head_block_size * head_dim
    triton_block_size = head_dim
    assert per_token_size % triton_block_size == 0
    cu_src_ptrs = torch.tensor(src_ptrs, dtype=torch.int64, device="cuda")
    cu_dst_ptrs = torch.tensor(dst_ptrs, dtype=torch.int64, device="cuda")
    return (
        [src_buffer, dst_buffer],
        cu_src_ptrs,
        cu_dst_ptrs,
        n_tokens,
        block_size,
        per_token_size,
        triton_block_size,
    )


def _gen_sum_inputs(
    n_heads,
    head_dim,
    block_size: int,
    max_n_ops_per_block: int,
    n_reds: int,
    triton_block_size: int,
    n_blocks: int,
    dtype: torch.dtype,
):
    src_buffer = torch.randn(
        n_blocks, block_size, n_heads, head_dim, dtype=dtype, device="cuda"
    )
    dst_buffer = torch.randn_like(src_buffer)

    ops_per_red = torch.full(
        (n_reds,), max_n_ops_per_block, dtype=torch.int32, device="cuda"
    )
    red_blocks = np.random.choice(
        n_blocks, ops_per_red.sum().item(), replace=False
    ).tolist()
    src_offsets = []
    src_offsets_list = []
    cumu_ops = 0
    for n_ops in ops_per_red:
        src_offsets.append(
            red_blocks[cumu_ops : cumu_ops + n_ops]
            + [0 for _ in range(max_n_ops_per_block - n_ops)]
        )
        src_offsets_list.append(red_blocks[cumu_ops : cumu_ops + n_ops])
        cumu_ops += n_ops
    src_offsets = torch.tensor(src_offsets, dtype=torch.int32, device="cuda")

    n_elems = (
        torch.randint(
            1, block_size + 1, (n_reds,), dtype=torch.int32, device="cuda"
        )
        * n_heads
        * head_dim
    )

    src_ptrs = torch.tensor(
        [src_buffer.data_ptr() for _ in range(n_reds)],
        dtype=torch.int64,
        device="cuda",
    )
    src_strides = torch.tensor(
        [src_buffer.stride(1) for _ in range(n_reds)],
        dtype=torch.int32,
        device="cuda",
    )
    dst_ptrs = torch.tensor(
        [dst_buffer.data_ptr() for _ in range(n_reds)],
        dtype=torch.int64,
        device="cuda",
    )
    dst_strides = torch.tensor(
        [dst_buffer.stride(1) for _ in range(n_reds)],
        dtype=torch.int32,
        device="cuda",
    )
    dst_offsets = np.random.choice(n_blocks, n_reds, replace=False).tolist()
    dst_offsets = torch.tensor(dst_offsets, dtype=torch.int32, device="cuda")
    return (
        [src_buffer, dst_buffer],
        src_ptrs,
        src_offsets,
        src_strides,
        ops_per_red,
        dst_ptrs,
        dst_offsets,
        dst_strides,
        n_elems,
        dtype,
        max_n_ops_per_block,
        block_size * n_heads * head_dim,
        triton_block_size,
    )


def warmup_triton_ops(
    q_head_block_size,
    kv_head_block_size,
    head_dim,
    main_dtype: torch.dtype,
    flash_attn_update_triton_block_size: int = 64,
    memcpy_triton_block_size: int = 128,
    sum_triton_block_size: int = 128,
    verbose: bool = False,
):
    if verbose:
        import tqdm
    if "L40" in torch.cuda.get_device_name():
        flash_attn_update_triton_block_size = 32
    # warmup _flash_attn_update_out_lse
    _max_n_blocks_per_ops = [1, 2, 4, 8, 16, 32, 64]
    if verbose:
        pbar = tqdm.tqdm(total=len(_max_n_blocks_per_ops))
    for max_n_blocks_per_op in _max_n_blocks_per_ops:
        if verbose:
            pbar.set_description(
                f"Warmup flash_attn_update_out_lse max_n_blocks_per_op {max_n_blocks_per_op}"
            )
        args = _gen_flash_attn_update_out_lse_inputs(
            q_head_block_size,
            head_dim,
            max_n_blocks_per_op,
            1024,
            flash_attn_update_triton_block_size,
            main_dtype,
        )
        flash_attn_update_out_lse(*args)
        if verbose:
            pbar.update(1)
    # warmup copy ops
    _block_size = [128, 256, 512, 1024, 2048, 4096, 8192]
    dtype_to_copy_fns = {
        torch.float32: fused_copy_fp32_varlen,
        torch.float16: fused_copy_fp16_varlen,
        torch.bfloat16: fused_copy_bf16_varlen,
    }
    if verbose:
        pbar = tqdm.tqdm(total=len(_block_size) * len(dtype_to_copy_fns))
    for dtype, copy_fn in dtype_to_copy_fns.items():
        for block_size in _block_size:
            if verbose:
                pbar.set_description(
                    f"Warmup memcpy dype {dtype}, block_size {block_size}"
                )
            args = _gen_memcpy_inputs(
                block_size, memcpy_triton_block_size, 8, dtype
            )
            copy_fn(*args[1:])
            if verbose:
                pbar.update(1)
    dtype_to_copy_fns_lse = {
        torch.float32: fused_copy_fp32_varlen_lse,
    }
    if verbose:
        pbar = tqdm.tqdm(total=len(_block_size) * len(dtype_to_copy_fns_lse))
    for dtype, copy_fn in dtype_to_copy_fns_lse.items():
        for block_size in _block_size:
            if verbose:
                pbar.set_description(
                    f"Warmup memcpy_lse dype {dtype}, block_size {block_size}"
                )
            args = _gen_memcpy_lse_inputs(
                block_size, memcpy_triton_block_size, 8, dtype
            )
            copy_fn(*args[1:])
            if verbose:
                pbar.update(1)
    dtype_to_dkv_copy_fns = {
        torch.float32: fused_copy_fp32_varlen_dkv,
        torch.float16: fused_copy_fp16_varlen_dkv,
        torch.bfloat16: fused_copy_bf16_varlen_dkv,
    }
    if verbose:
        pbar = tqdm.tqdm(total=len(_block_size))
    for block_size in _block_size:
        if verbose:
            pbar.set_description(f"Warmup memcpy_dkv block_size {block_size}")
        args = _gen_memcpy_dkv_inputs(
            block_size, kv_head_block_size, head_dim, main_dtype
        )
        dtype_to_dkv_copy_fns[main_dtype](*args[1:])
        if verbose:
            pbar.update(1)
    # warmup sum ops
    _max_n_ops_per_block = [1, 2, 4, 8, 16, 32, 64]
    if verbose:
        pbar = tqdm.tqdm(
            total=len(_max_n_ops_per_block) * len(_block_size) * 3
        )
    torch.random.manual_seed(24)
    np.random.seed(24)
    for block_size in _block_size:
        for max_n_ops_per_block in _max_n_ops_per_block:
            for dtype in [torch.float32, torch.float16, torch.bfloat16]:
                if verbose:
                    pbar.set_description(
                        f"Warmup sum block_size {block_size}, max_n_ops_per_block {max_n_ops_per_block}, dtype {dtype}"
                    )
                args = _gen_sum_inputs(
                    q_head_block_size,
                    head_dim,
                    block_size,
                    max_n_ops_per_block,
                    2,
                    sum_triton_block_size,
                    128,
                    dtype,
                )
                fused_blockwise_sum(*args[1:])
                # also warmup for kv head block size
                args = _gen_sum_inputs(
                    kv_head_block_size,
                    head_dim,
                    block_size,
                    max_n_ops_per_block,
                    2,
                    sum_triton_block_size,
                    128,
                    dtype,
                )
                fused_blockwise_sum(*args[1:])
                if verbose:
                    pbar.update(1)
