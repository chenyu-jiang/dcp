from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from dcp.ops.fused_ops import _flash_attn_update_out_lse


# modified from https://github.com/zhuzilin/ring-flash-attention/blob/main/ring_flash_attn/utils.py
def _update_out_and_lse(
    out: torch.Tensor,
    lse: torch.Tensor,
    block_out: torch.Tensor,
    block_lse: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:

    block_out = block_out.to(torch.float32)
    block_lse = block_lse.transpose(-2, -1).unsqueeze(dim=-1)
    lse = lse.transpose(-2, -1).unsqueeze(dim=-1)

    new_lse = lse + torch.log(1 + torch.exp(block_lse - lse))
    new_out = (
        torch.exp(lse - new_lse) * out
        + torch.exp(block_lse - new_lse) * block_out
    )

    return new_out, new_lse.squeeze().transpose(-2, -1)


def ref_update_out_lse(
    src_out_buffer: torch.Tensor,
    dst_out_buffer: torch.Tensor,
    src_lse_buffer: torch.Tensor,
    dst_lse_buffer: torch.Tensor,
    src_block_table: torch.Tensor,
    dst_token_offsets: torch.Tensor,
    block_lengths: torch.Tensor,
    n_tokens_per_block: torch.Tensor,
):
    n_ops = len(block_lengths)
    for op_id in range(n_ops):
        src_block_indices = src_block_table[
            op_id, : block_lengths[op_id].item()
        ].tolist()
        dst_block_index = dst_token_offsets[op_id].item()
        for block_idx, src_indice in enumerate(src_block_indices):
            n_tokens = n_tokens_per_block[op_id].item()
            src_out_block = src_out_buffer[
                src_indice,
                :n_tokens,
            ]  # [n_tokens, n_head, head_dim]
            dst_out_block = dst_out_buffer[
                dst_block_index : dst_block_index + n_tokens,
            ]  # [n_tokens, n_head, head_dim]

            src_lse_block = src_lse_buffer[
                :,
                src_indice,
                :n_tokens,
            ]  # [n_head, n_tokens]
            dst_lse_block = dst_lse_buffer[
                :,
                dst_block_index : dst_block_index + n_tokens,
            ]  # [n_head, n_tokens]
            if block_idx == 0:
                (
                    dst_out_buffer[
                        dst_block_index : dst_block_index + n_tokens
                    ],
                    dst_lse_buffer[
                        :, dst_block_index : dst_block_index + n_tokens
                    ],
                ) = (src_out_block, src_lse_block)
            else:
                (
                    dst_out_buffer[
                        dst_block_index : dst_block_index + n_tokens
                    ],
                    dst_lse_buffer[
                        :, dst_block_index : dst_block_index + n_tokens
                    ],
                ) = _update_out_and_lse(
                    dst_out_block, dst_lse_block, src_out_block, src_lse_block
                )


def _round_to_pow_2(x: int):
    return 1 << (x - 1).bit_length()


def test_flash_attn_update_out_lse(
    n_src_blocks_per_op: List[int],
    src_buffer_size_n_blocks: int,
    dst_buffer_size_n_blocks: int,
    n_head: int,
    head_dim: int,
    triton_block_size: int,
):
    torch.set_default_device("cuda")
    assert sum(n_src_blocks_per_op) <= src_buffer_size_n_blocks
    assert len(n_src_blocks_per_op) <= dst_buffer_size_n_blocks
    with torch.no_grad():
        block_size = 256
        n_tokens_per_block = torch.randint(
            1, block_size, (len(n_src_blocks_per_op),), dtype=torch.int32
        )

        dst_token_offsets = F.pad(n_tokens_per_block, (1, 0), value=0).cumsum(
            0
        )

        src_out_buffer = torch.randn(
            src_buffer_size_n_blocks, block_size, n_head, head_dim
        )
        dst_out_buffer = torch.zeros(
            torch.sum(n_tokens_per_block).item(), n_head, head_dim
        )
        src_lse_buffer = torch.randn(
            n_head, src_buffer_size_n_blocks, block_size
        )
        dst_lse_buffer = torch.zeros(
            n_head, torch.sum(n_tokens_per_block).item()
        )

        src_block_table_combined = torch.tensor(
            np.random.choice(
                src_buffer_size_n_blocks,
                ((sum(n_src_blocks_per_op),)),
                replace=False,
            ),
            dtype=torch.int32,
        ).tolist()

        max_n_src_blocks_per_op = _round_to_pow_2(max(n_src_blocks_per_op))

        src_block_table = []
        c = 0
        for n in n_src_blocks_per_op:
            curr_op_src_blocks = src_block_table_combined[c : c + n] + [0] * (
                max_n_src_blocks_per_op - n
            )
            src_block_table.append(curr_op_src_blocks)
            c += n
        src_block_table = torch.tensor(src_block_table, dtype=torch.int32)

        def _get_grid_dims():
            assert block_size % triton_block_size == 0
            n_programs_per_block = block_size // triton_block_size
            n_red_ops = len(n_src_blocks_per_op)
            return (n_red_ops * n_head * n_programs_per_block,)

        block_lengths_tensor = torch.tensor(
            n_src_blocks_per_op, dtype=torch.int32
        )

        ref_dst_out_buffer = dst_out_buffer.clone()
        ref_dst_lse_buffer = dst_lse_buffer.clone()

        grid_dims = _get_grid_dims()
        _flash_attn_update_out_lse[grid_dims](
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
            max_n_src_blocks_per_op,
            triton_block_size,
        )

        ref_update_out_lse(
            src_out_buffer,
            ref_dst_out_buffer,
            src_lse_buffer,
            ref_dst_lse_buffer,
            src_block_table,
            dst_token_offsets,
            block_lengths_tensor,
            n_tokens_per_block,
        )

        print(
            "Max diff dst_out_buffer",
            torch.max(torch.abs(dst_out_buffer - ref_dst_out_buffer)),
        )
        print(
            "Mean diff dst_out_buffer",
            torch.mean(torch.abs(dst_out_buffer - ref_dst_out_buffer)),
        )
        print(
            "Max diff dst_lse_buffer",
            torch.max(torch.abs(dst_lse_buffer - ref_dst_lse_buffer)),
        )
        print(
            "Mean diff dst_lse_buffer",
            torch.mean(torch.abs(dst_lse_buffer - ref_dst_lse_buffer)),
        )


if __name__ == "__main__":
    test_flash_attn_update_out_lse(
        n_src_blocks_per_op=[2, 4, 4, 2, 5],
        src_buffer_size_n_blocks=24,
        dst_buffer_size_n_blocks=18,
        n_head=4,
        head_dim=64,
        triton_block_size=128,
    )
