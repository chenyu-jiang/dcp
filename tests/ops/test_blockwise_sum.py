from typing import List

import numpy as np
import torch

from dcp.ops.fused_ops import fused_blockwise_sum


def ref_blockwise_sum(
    src_buffer: torch.Tensor,
    src_indices: List[List[int]],
    dst_buffer: torch.Tensor,
    dst_indices: List[int],
    n_elems: List[int],
):
    for src_offsets, dst_offset, n_elem in zip(
        src_indices, dst_indices, n_elems
    ):
        src_tensor = src_buffer[src_offsets, :n_elem]
        red_result = torch.sum(src_tensor, dim=0, dtype=torch.float32).to(
            dst_buffer.dtype
        )
        dst_buffer[dst_offset, :n_elem] = red_result


def test_fused_blockwise_sum(dtype=torch.bfloat16):
    torch.set_default_dtype(dtype)
    torch.set_default_device("cuda")
    block_size = 512
    n_blocks = 64
    n_heads = 2
    head_dim = 64
    max_n_ops_per_block = 8
    n_reds = 8

    src_buffer = torch.randn(
        n_blocks, block_size, n_heads, head_dim, dtype=dtype
    )
    dst_buffer = torch.randn_like(src_buffer)

    ops_per_red = torch.randint(
        1, max_n_ops_per_block + 1, (n_reds,), dtype=torch.int32
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
    src_offsets = torch.tensor(src_offsets, dtype=torch.int32)

    n_elems = (
        torch.randint(1, block_size + 1, (n_reds,), dtype=torch.int32)
        * n_heads
        * head_dim
    )

    src_ptrs = torch.tensor(
        [src_buffer.data_ptr() for _ in range(n_reds)], dtype=torch.int64
    )
    src_strides = torch.tensor(
        [src_buffer.stride(0) for _ in range(n_reds)], dtype=torch.int32
    )
    dst_ptrs = torch.tensor(
        [dst_buffer.data_ptr() for _ in range(n_reds)], dtype=torch.int64
    )
    dst_strides = torch.tensor(
        [dst_buffer.stride(0) for _ in range(n_reds)], dtype=torch.int32
    )
    dst_offsets = np.random.choice(n_blocks, n_reds, replace=False).tolist()
    dst_offsets = torch.tensor(dst_offsets, dtype=torch.int32)

    ref_dst_buffer = dst_buffer.clone()
    fused_blockwise_sum(
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
        128,
    )
    ref_blockwise_sum(
        src_buffer,
        src_offsets_list,
        ref_dst_buffer,
        dst_offsets.tolist(),
        [x // n_heads // head_dim for x in n_elems.tolist()],
    )
    assert torch.allclose(ref_dst_buffer, dst_buffer, atol=1e-2)


if __name__ == "__main__":
    # test_fused_blockwise_sum(torch.bfloat16)
    # test_fused_blockwise_sum(torch.float32)
    test_fused_blockwise_sum(torch.float16)
    print("Tests passed.")
