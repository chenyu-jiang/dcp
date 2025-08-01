from dcp.runtime.flash_attention.executor import (
    _wrapped_flash_attn_varlen_forward,
)

import torch

(
    buffer_q,
    buffer_kv,
    cu_seqlens_q,
    cu_seqlens_kv,
    total_q,
    max_seqlen_q,
    max_seqlen_kv,
    attn_mask,
    cu_q_block_table,
    cu_kv_block_table,
    cu_out_block_table,
    buffer_out,
    buffer_lse,
) = torch.load("/root/.dcp_debug/Attn-D0-I4.pt")

print(
    "====================================BEFORE===================================="
)

print("attention_mask shape: ", attn_mask.shape)
print("attention_mask[:512]: ")
print(attn_mask[:, :, :512])
print("attention_mask[512:]: ")
print(attn_mask[:, :, 512 : 512 + 26])

print("----")
print("cu_out_block_table: ", cu_out_block_table)


def _get_attn_mask_valid_range(attn_mask):
    attn_range1_start = attn_mask[0, 0, :]
    attn_range1_end = attn_mask[0, 1, :]
    attn_range2_start = attn_mask[1, 0, :]
    attn_range2_end = attn_mask[1, 1, :]
    attn_valid_range1 = attn_range1_start < attn_range1_end
    attn_valid_range2 = attn_range2_start < attn_range2_end
    attn_valid_range = torch.logical_or(attn_valid_range1, attn_valid_range2)
    return attn_valid_range


_wrapped_flash_attn_varlen_forward(
    buffer_q,
    buffer_kv,
    cu_seqlens_q,
    cu_seqlens_kv,
    total_q,
    max_seqlen_q,
    max_seqlen_kv,
    attn_mask,
    cu_q_block_table,
    cu_kv_block_table,
    cu_out_block_table,
    buffer_out,
    buffer_lse,
)

print(
    "====================================AFTER===================================="
)
for i in range(buffer_lse.shape[1]):
    print("Buffer LSE Block", i)
    print(buffer_lse[0, i, :])

# concat out buffer lse blocks
out_block_table = cu_out_block_table.cpu().tolist()
concated_lse = []
for i in range(len(out_block_table)):
    concated_lse.append(buffer_lse[0, out_block_table[i], :].squeeze())
concated_lse = torch.concat(concated_lse)

valid_attn_range = _get_attn_mask_valid_range(attn_mask)

masked_lse = concated_lse[_get_attn_mask_valid_range(attn_mask)]
assert (masked_lse > 0).all() and masked_lse.isfinite().all()
