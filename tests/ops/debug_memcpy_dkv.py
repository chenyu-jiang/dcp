import torch

from dcp.ops.fused_ops import (
    fused_copy_bf16_varlen_dkv,
    _gen_memcpy_dkv_inputs,
)

(
    [src_buffer, dst_buffer],
    cu_src_ptrs,
    cu_dst_ptrs,
    cu_n_tokens,
    block_size,
    per_token_size,
    triton_block_size,
) = _gen_memcpy_dkv_inputs(128, 1, 128, dtype=torch.bfloat16)

n_progs = len(cu_src_ptrs) * block_size * (per_token_size // triton_block_size)

fused_copy_bf16_varlen_dkv(
    cu_src_ptrs,
    cu_dst_ptrs,
    cu_n_tokens,
    block_size,
    per_token_size,
    triton_block_size,
)

torch.cuda.synchronize()
