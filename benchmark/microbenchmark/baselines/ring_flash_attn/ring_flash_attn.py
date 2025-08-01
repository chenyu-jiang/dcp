import torch
import torch.distributed as dist
import torch.nn.functional as F
from ring_flash_attn import (
    ring_flash_attn_varlen_kvpacked_func,
    zigzag_ring_flash_attn_varlen_kvpacked_func,
)
from utils import extract_local_ring, extract_local_zigzag

_CURR_RING_GROUP = None


def prepare_zigzag_ring_attn(config, seqlens, q, kv, dout):
    global _CURR_RING_GROUP
    if _CURR_RING_GROUP is None:
        _CURR_RING_GROUP = config.cp_group
    ring_group = _CURR_RING_GROUP
    world_size = dist.get_world_size(group=ring_group)
    cu_seqlens = F.pad(
        torch.cumsum(torch.tensor(seqlens, dtype=torch.int32), 0),
        (1, 0),
    ).to(torch.int32)
    assert torch.all(cu_seqlens % (2 * world_size) == 0)
    max_seqlen = max(seqlens)
    local_q = extract_local_zigzag(
        q, cu_seqlens, dist.get_rank(group=ring_group), world_size
    )
    local_kv = extract_local_zigzag(
        kv, cu_seqlens, dist.get_rank(group=ring_group), world_size
    )
    local_dout = extract_local_zigzag(
        dout, cu_seqlens, dist.get_rank(group=ring_group), world_size
    )
    local_cu_seqlens = cu_seqlens // world_size
    local_max_seqlen = max_seqlen // world_size
    return local_q, local_kv, local_dout, local_cu_seqlens, local_max_seqlen


def prepare_ring_attn(config, seqlens, q, kv, dout):
    global _CURR_RING_GROUP
    if _CURR_RING_GROUP is None:
        _CURR_RING_GROUP = config.cp_group
    ring_group = _CURR_RING_GROUP
    world_size = dist.get_world_size(group=ring_group)
    cu_seqlens = F.pad(
        torch.cumsum(torch.tensor(seqlens, dtype=torch.int32), 0),
        (1, 0),
    ).to(torch.int32)
    assert torch.all(cu_seqlens % world_size == 0)
    max_seqlen = max(seqlens)
    local_q = extract_local_ring(
        q, cu_seqlens, dist.get_rank(group=ring_group), world_size
    )
    local_kv = extract_local_ring(
        kv, cu_seqlens, dist.get_rank(group=ring_group), world_size
    )
    local_dout = extract_local_ring(
        dout, cu_seqlens, dist.get_rank(group=ring_group), world_size
    )
    local_cu_seqlens = cu_seqlens // world_size
    local_max_seqlen = max_seqlen // world_size
    return local_q, local_kv, local_dout, local_cu_seqlens, local_max_seqlen
