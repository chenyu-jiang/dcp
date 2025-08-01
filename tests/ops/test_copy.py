from typing import List

import numpy as np
import torch

from dcp.ops.fused_ops import (
    _fused_copy_bf16,
    _fused_copy_fp32,
    _fused_copy_bf16_varlen,
    _fused_copy_bf16_varlen_dkv,
    _fused_copy_fp32_varlen_lse,
)


def generate_blocks(n_blocks_per_size: List[int], block_sizes: List[int]):
    buffers = []
    dst_buffers = []

    src_ptrs = []
    dst_ptrs = []
    src_offsets = []
    dst_offsets = []
    src_strides = []
    dst_strides = []
    n_elements = []
    for block_size, n_blocks in zip(block_sizes, n_blocks_per_size):
        buffer = torch.randn(n_blocks * 2, block_size)
        buffers.append(buffer)
        dst_buffer = torch.randn_like(buffer)
        dst_buffers.append(dst_buffer)
        buffer_ptrs = torch.tensor(
            [buffer.data_ptr() for _ in range(n_blocks)]
        )
        dst_buffer_ptrs = torch.tensor(
            [dst_buffer.data_ptr() for _ in range(n_blocks)]
        )
        src_ptrs.append(buffer_ptrs)
        dst_ptrs.append(dst_buffer_ptrs)
        block_offsets = torch.tensor(
            np.random.choice(n_blocks * 2, n_blocks, replace=False),
            dtype=torch.int32,
        )
        dst_block_offsets = torch.tensor(
            np.random.choice(n_blocks * 2, n_blocks, replace=False),
            dtype=torch.int32,
        )
        block_strides = torch.tensor(
            [block_size for _ in range(n_blocks)], dtype=torch.int32
        )
        dst_block_strides = torch.tensor(
            [block_size for _ in range(n_blocks)], dtype=torch.int32
        )
        src_offsets.append(block_offsets)
        dst_offsets.append(dst_block_offsets)
        src_strides.append(block_strides)
        dst_strides.append(dst_block_strides)
        n_elements.append(
            torch.randint(1, block_size + 1, (n_blocks,), dtype=torch.int32)
        )
    return (
        src_ptrs,
        src_offsets,
        src_strides,
        buffers,
        dst_ptrs,
        dst_offsets,
        dst_strides,
        dst_buffers,
        n_elements,
    )


def ref_copy(
    src_buffers: List[torch.Tensor],
    src_offsets: List[torch.Tensor],
    dst_buffers: List[torch.Tensor],
    dst_offsets: List[torch.Tensor],
):
    for src_buffer, src_block_offsets, dst_buffer, dst_block_offsets in zip(
        src_buffers, src_offsets, dst_buffers, dst_offsets
    ):
        dst_buffer[dst_block_offsets] = src_buffer[src_block_offsets]


def ref_copy_varlen(
    src_buffers: List[torch.Tensor],
    src_offsets: List[torch.Tensor],
    dst_buffers: List[torch.Tensor],
    dst_offsets: List[torch.Tensor],
    n_elements: List[torch.Tensor],
):
    for (
        src_buffer,
        src_block_offsets,
        dst_buffer,
        dst_block_offsets,
        n_element,
    ) in zip(src_buffers, src_offsets, dst_buffers, dst_offsets, n_elements):
        for src_block_offset, dst_block_offset, n_element in zip(
            src_block_offsets, dst_block_offsets, n_element
        ):
            dst_buffer[dst_block_offset, :n_element] = src_buffer[
                src_block_offset, :n_element
            ]


def ref_copy_varlen_lse(
    src_buffer: torch.Tensor,
    src_offsets: List[int],
    dst_buffer: torch.Tensor,
    dst_offsets: List[int],
    n_elements: List[int],
):
    for src_block_offset, dst_block_offset, n_element in zip(
        src_offsets, dst_offsets, n_elements
    ):
        dst_buffer[:, dst_block_offset : dst_block_offset + n_element] = (
            src_buffer[:, src_block_offset : src_block_offset + n_element]
        )


def test_fused_copy(dtype=torch.float32):
    torch.set_default_dtype(dtype)
    torch.set_default_device("cuda")
    block_sizes = [256 * 128, 512 * 128, 1024 * 128]
    n_blocks_per_size = [2, 3, 4]
    (
        src_ptrs,
        src_offsets,
        src_strides,
        buffers,
        dst_ptrs,
        dst_offsets,
        dst_strides,
        dst_buffers,
        _,
    ) = generate_blocks(n_blocks_per_size, block_sizes)
    ref_dst_buffers = [buffer.clone() for buffer in dst_buffers]
    ref_copy(buffers, src_offsets, ref_dst_buffers, dst_offsets)

    triton_block_size = 256
    for (
        block_size,
        src_buffer_ptrs,
        src_block_offsets,
        src_block_strides,
        dst_buffer_ptrs,
        dst_block_offsets,
        dst_block_strides,
    ) in zip(
        block_sizes,
        src_ptrs,
        src_offsets,
        src_strides,
        dst_ptrs,
        dst_offsets,
        dst_strides,
    ):
        n_programs_per_block = block_size // triton_block_size
        copy_fn = (
            _fused_copy_bf16 if dtype == torch.bfloat16 else _fused_copy_fp32
        )
        copy_fn[(len(src_buffer_ptrs) * n_programs_per_block,)](
            src_buffer_ptrs,
            src_block_offsets,
            src_block_strides,
            dst_buffer_ptrs,
            dst_block_offsets,
            dst_block_strides,
            block_size,
            triton_block_size,
        )
    for ref_dst_buffer, dst_buffer, dst_block_offsets in zip(
        ref_dst_buffers, dst_buffers, dst_offsets
    ):
        assert torch.allclose(ref_dst_buffer, dst_buffer)


def test_fused_copy_varlen(dtype=torch.bfloat16):
    torch.set_default_dtype(dtype)
    torch.set_default_device("cuda")
    block_sizes = [256 * 128, 512 * 128, 1024 * 128]
    n_blocks_per_size = [2, 3, 4]
    (
        src_ptrs,
        src_offsets,
        src_strides,
        buffers,
        dst_ptrs,
        dst_offsets,
        dst_strides,
        dst_buffers,
        n_elements,
    ) = generate_blocks(n_blocks_per_size, block_sizes)
    ref_dst_buffers = [buffer.clone() for buffer in dst_buffers]
    ref_copy_varlen(
        buffers, src_offsets, ref_dst_buffers, dst_offsets, n_elements
    )

    triton_block_size = 256
    for (
        block_size,
        src_buffer_ptrs,
        src_block_offsets,
        src_block_strides,
        dst_buffer_ptrs,
        dst_block_offsets,
        dst_block_strides,
        n_element,
    ) in zip(
        block_sizes,
        src_ptrs,
        src_offsets,
        src_strides,
        dst_ptrs,
        dst_offsets,
        dst_strides,
        n_elements,
    ):
        n_programs_per_block = block_size // triton_block_size
        copy_fn = _fused_copy_bf16_varlen
        copy_fn[(len(src_buffer_ptrs) * n_programs_per_block,)](
            src_buffer_ptrs,
            src_block_offsets,
            src_block_strides,
            dst_buffer_ptrs,
            dst_block_offsets,
            dst_block_strides,
            n_element,
            block_size,
            triton_block_size,
        )
    for ref_dst_buffer, dst_buffer, dst_block_offsets in zip(
        ref_dst_buffers, dst_buffers, dst_offsets
    ):
        assert torch.allclose(ref_dst_buffer, dst_buffer)


def test_fused_copy_varlen_lse(dtype=torch.float32):
    torch.set_default_dtype(dtype)
    torch.set_default_device("cuda")
    head_block_size = 8
    n_tokens = 4096
    block_size = 512
    n_blocks = 4
    triton_block_size = 128
    src_buffer = torch.randn(head_block_size, n_tokens)
    dst_buffer = torch.randn_like(src_buffer)
    src_offsets = torch.randint(0, block_size, (n_blocks,), dtype=torch.int32)
    src_offsets += torch.arange(0, n_blocks) * (n_tokens // n_blocks)
    n_tokens_tensor = torch.randint(
        1, block_size + 1, (n_blocks,), dtype=torch.int32
    )
    dst_offsets = torch.randint(0, block_size, (n_blocks,), dtype=torch.int32)
    dst_offsets += torch.arange(0, n_blocks) * (n_tokens // n_blocks)
    copy_fn = _fused_copy_fp32_varlen_lse

    src_ptrs = torch.tensor([src_buffer.data_ptr() for _ in range(n_blocks)])
    dst_ptrs = torch.tensor([dst_buffer.data_ptr() for _ in range(n_blocks)])
    head_strides = torch.tensor(
        [n_tokens for _ in range(n_blocks)], dtype=torch.int32
    )

    ref_dst_buffer = dst_buffer.clone()

    n_programs_per_block = block_size // triton_block_size
    copy_fn[(n_blocks * n_programs_per_block,)](
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
    ref_copy_varlen_lse(
        src_buffer,
        src_offsets.tolist(),
        ref_dst_buffer,
        dst_offsets.tolist(),
        n_tokens_tensor.tolist(),
    )
    assert torch.allclose(ref_dst_buffer, dst_buffer)


def test_copy_varlen_dkv():
    dtype = torch.bfloat16
    torch.set_default_dtype(dtype)
    torch.set_default_device("cuda")
    n_heads = 1
    head_dim = 128
    max_n_tokens = 4096

    max_n_blocks = 8
    src_block_indices = [3, 6, 2, 0]
    dst_block_indices = [7, 1, 5, 4]

    src_buffer = torch.randn(max_n_blocks, max_n_tokens, 2, n_heads, head_dim)
    dst_buffer = torch.zeros_like(src_buffer)

    dst_buffer_ref = torch.zeros_like(src_buffer)

    src_ptrs = [src_buffer[idx].data_ptr() for idx in src_block_indices]
    dst_ptrs = [dst_buffer[idx].data_ptr() for idx in dst_block_indices]
    n_tokens = [126, 287, 520, 3028]
    per_token_size = 2 * n_heads * head_dim
    n_ptrs = len(src_ptrs)
    triton_block_size = head_dim
    assert per_token_size % triton_block_size == 0
    n_progs = n_ptrs * max_n_tokens * (per_token_size // triton_block_size)
    cu_src_ptrs = torch.tensor(src_ptrs, dtype=torch.int64)
    cu_dst_ptrs = torch.tensor(dst_ptrs, dtype=torch.int64)
    cu_n_tokens = torch.tensor(n_tokens, dtype=torch.int32)
    _fused_copy_bf16_varlen_dkv[(n_progs,)](
        cu_src_ptrs,
        cu_dst_ptrs,
        cu_n_tokens,
        max_n_tokens,
        per_token_size,
        triton_block_size,
    )
    for src_idx, dst_idx, n_token in zip(
        src_block_indices, dst_block_indices, n_tokens
    ):
        dst_buffer_ref[dst_idx, :n_token] = src_buffer[src_idx, :n_token]
    assert torch.allclose(dst_buffer, dst_buffer_ref)


if __name__ == "__main__":
    # test_fused_copy(torch.bfloat16)
    # test_fused_copy_varlen(torch.bfloat16)
    # test_fused_copy_varlen_lse(torch.float32)
    test_copy_varlen_dkv()
