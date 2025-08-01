from typing import List
import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch.multiprocessing as mp
import transformer_engine as te
from baselines.common import create_groups_with_cp_ranks
from utils import extract_local_zigzag

from dcp.runtime.flash_attention.utils import (
    lambda_mask_fn,
    shared_question_mask_fn,
    causal_blockwise_mask_fn,
    get_local_attn_ranges_zigzag,
    get_per_step_attn_range,
)

_TE_ATTN_MODULE = None
_TE_CP_GROUP = None


def _get_mask_fn(mask_type: str):
    if mask_type == "causal":
        return None
    elif mask_type == "lambda":
        return lambda_mask_fn
    elif mask_type == "shared_question":
        return shared_question_mask_fn
    elif mask_type == "causal_blockwise":
        return causal_blockwise_mask_fn
    else:
        raise ValueError(f"Unsupported mask type: {mask_type}")


def _init_te_attn_module(config):
    global _TE_ATTN_MODULE
    global _TE_CP_GROUP
    if config.mask == "causal":
        te_mask_type = "causal"
    else:
        te_mask_type = "custom_ranges"
    world_size = dist.get_world_size(group=config.cp_group)
    if config.te_cp_comm_type == "a2a+p2p":
        # create subgroups for a2a and p2p separately
        if config.a2a_degree == -1:
            head_degree = min(config.n_query_groups, world_size)
        else:
            head_degree = config.a2a_degree
        assert (
            world_size % head_degree == 0
        ), "world size must be divisible by head_degree"
        ring_degree = world_size // head_degree
    elif config.te_cp_comm_type == "a2a":
        ring_degree = 1
        head_degree = world_size
    else:
        ring_degree = world_size
        head_degree = 1
    if config.p2p_a2a_order == "AP":
        head_ranks = [
            list(range(i * head_degree, (i + 1) * head_degree))
            for i in range(ring_degree)
        ]
        ring_ranks = [
            list(range(i, world_size, head_degree)) for i in range(head_degree)
        ]
    elif config.p2p_a2a_order == "PA":
        raise NotImplementedError("PA order is not supported in TE.")
    head_groups = create_groups_with_cp_ranks(config, head_ranks)
    ring_groups = create_groups_with_cp_ranks(config, ring_ranks)
    local_head_group = head_groups[
        dist.get_rank(group=config.cp_group) // head_degree
    ]
    local_ring_group = ring_groups[
        dist.get_rank(group=config.cp_group) % head_degree
    ]
    if config.te_cp_comm_type == "a2a+p2p":
        cp_group = [local_head_group, local_ring_group]
    else:
        cp_group = config.cp_group

    te_attn_module = te.pytorch.DotProductAttention(
        num_attention_heads=config.n_heads,
        kv_channels=config.head_dim,  # same as head_dim
        num_gqa_groups=config.n_query_groups,
        attn_mask_type=te_mask_type,
        qkv_format="thd",  # if config.te_cp_comm_type == "p2p" else "bshd",
        cp_group=cp_group,
        cp_global_ranks=config.curr_cpg_ranks,
        cp_stream=torch.cuda.Stream(),
        cp_comm_type=config.te_cp_comm_type,
    )
    _TE_ATTN_MODULE = te_attn_module
    _TE_CP_GROUP = cp_group


def prepare_te_attn(
    config, seqlens, q: torch.Tensor, kv: torch.Tensor, dout: torch.Tensor
):
    world_size = dist.get_world_size(group=config.cp_group)
    if _TE_ATTN_MODULE is None:
        _init_te_attn_module(config)
    te_attn_module = _TE_ATTN_MODULE
    cu_seqlens = F.pad(
        torch.cumsum(torch.tensor(seqlens, dtype=torch.int32), 0), (1, 0)
    ).to(torch.int32)
    assert torch.all(cu_seqlens % (2 * world_size) == 0)
    max_seqlen = max(seqlens)
    local_q = extract_local_zigzag(
        q, cu_seqlens, dist.get_rank(group=config.cp_group), world_size
    )
    local_kv = extract_local_zigzag(
        kv, cu_seqlens, dist.get_rank(group=config.cp_group), world_size
    )
    local_dout = extract_local_zigzag(
        dout, cu_seqlens, dist.get_rank(group=config.cp_group), world_size
    )
    cu_seqlens_cpu = cu_seqlens.cpu()
    return (
        te_attn_module,
        local_q,
        local_kv,
        local_dout,
        cu_seqlens,
        cu_seqlens_cpu,
        max_seqlen,
    )


def _te_mask_fn(t_args):
    torch.set_default_device("cpu")
    mask, seqlens, ring_world_size, ring_rank = t_args
    if mask == "causal":
        mask_fn = None
    else:
        mask_fn = _get_mask_fn(mask)
    if mask_fn is not None:
        attn_mask = mask_fn(seqlens, seqlens, torch.int32)
        if attn_mask.dim() == 3:
            attn_mask = attn_mask.permute(1, 2, 0).contiguous()
        else:
            attn_mask = attn_mask.transpose(1, 0).contiguous()

        attn_mask = get_local_attn_ranges_zigzag(
            attn_mask,
            seqlens,
            ring_world_size,
            ring_rank,
        ).cpu()
        cu_seqlens_cpu = F.pad(
            torch.cumsum(torch.tensor(seqlens, dtype=torch.int32), 0),
            (1, 0),
        ).cpu()
        per_step_attn_masks = [
            get_per_step_attn_range(
                i,
                ring_rank,
                ring_world_size,
                attn_mask,
                cu_seqlens_cpu,
            )
            for i in range(ring_world_size)
        ]
    else:
        per_step_attn_masks = None
    return per_step_attn_masks


def generate_masks_for_te(config, seqlens_per_batch: List[List[int]]):
    """
    Generate masks for transformer engine based on the config and seqlens.
    """
    if _TE_ATTN_MODULE is None:
        _init_te_attn_module(config)
    if isinstance(_TE_CP_GROUP, list):
        ring_group = _TE_CP_GROUP[1]
    else:
        ring_group = _TE_CP_GROUP
    ring_world_size = dist.get_world_size(group=ring_group)
    ring_rank = dist.get_rank(group=ring_group)

    with mp.Pool(processes=8) as pool:
        mask_generator = pool.imap(
            _te_mask_fn,
            (
                (config.mask, seqlens, ring_world_size, ring_rank)
                for seqlens in seqlens_per_batch
            ),
            chunksize=1,
        )
        yield from mask_generator
    pool.join()
    pool.close()
