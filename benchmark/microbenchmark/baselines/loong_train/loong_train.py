import torch
import torch.distributed as dist
import torch.nn.functional as F
from baselines.common import create_groups_with_cp_ranks
from baselines.loong_train.zigzag_ring_flash_attn_with_sliding_window import (
    zigzag_ring_flash_attn_kvpacked_func_with_sliding_window as loongtrain_attn_kvpacked_func,
)
from flash_attn.flash_attn_interface import (
    _flash_attn_backward,
    _flash_attn_forward,
)
from internlm.core.context import Config
from internlm.core.context.parallel_context import global_context as gpc
from internlm.core.parallel.comm.isp import _SeqAllToAll
from internlm.model.ops.utils import unpack_qkv_before_attn


def print_all(s):
    for rank in range(dist.get_world_size()):
        if rank == dist.get_rank():
            print(f"RANK {rank}: {s}")
        dist.barrier(device_ids=[torch.cuda.current_device()])


def print_0(s):
    if dist.get_rank() == 0:
        print(s)


# copied from InternEvo, strip away unnecessary
# internlm/core/parallel/comm/isp.py
class LTDistributedAttention(torch.nn.Module):
    """Initialization.

    Arguments:
        local_attention (Module): local self-attention module
        sequence_process_group (ProcessGroup): sequence parallel process group
    """

    def __init__(
        self,
        head_parallel_group: dist.ProcessGroup,
    ) -> None:
        super().__init__()
        self.hpg = head_parallel_group
        self.hp_size = dist.get_world_size(self.hpg)

    def forward(self, q, kv: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        # qkv shape: [1, packlen, 3, n_head, head_dim] or [batch, seqlen, 3, n_head, head_dim]
        # scatter in n_head and gather in seqlen(packlen)
        q, kv = _SeqAllToAll.apply(self.hpg, [2, 3], [1, 1], q, kv)

        context = loongtrain_attn_kvpacked_func(q, kv, *args, **kwargs)

        # context shape: [1, packlen, n_head, head_dim] or [batch, seqlen, n_head, head_dim]
        # scatter in seqlen(packlen) and gather in n_head
        context = _SeqAllToAll.apply(self.hpg, 1, 2, context)

        return context


_LT_ATTN_MOD = None
_LT_ATTN_KWARGS = None
_local_context_group = None
_local_head_group = None


def prepare_lt_attn(config, seqlens, q, kv, dout):
    # init process groups for loongtrain
    # Reference Impl:
    # Repo: https://github.com/InternLM/InternEvo.git
    # File: internlm/core/context/process_group_initializer.py
    global _LT_ATTN_MOD
    global _LT_ATTN_KWARGS
    global _local_context_group
    global _local_head_group
    world_size = dist.get_world_size(config.cp_group)
    assert (
        config.mask == "causal"
    ), "LoongTrain only support causal mask for now"
    if _LT_ATTN_MOD is None:
        if config.a2a_degree == -1:
            head_degree = min(config.n_query_groups, world_size)
        else:
            head_degree = config.a2a_degree
        assert (
            world_size % head_degree == 0
        ), "world size must be divisible by head_degree"
        ring_degree = world_size // head_degree
        if config.p2p_a2a_order == "AP":
            # equivalent to LoongTrain's head_first=True
            head_ranks = [
                list(range(i * head_degree, (i + 1) * head_degree))
                for i in range(ring_degree)
            ]
            ring_ranks = [
                list(range(i, world_size, head_degree))
                for i in range(head_degree)
            ]
        elif config.p2p_a2a_order == "PA":
            ring_ranks = [
                list(range(i * ring_degree, (i + 1) * ring_degree))
                for i in range(head_degree)
            ]
            head_ranks = [
                list(range(i, world_size, ring_degree))
                for i in range(ring_degree)
            ]
        head_groups = create_groups_with_cp_ranks(config, head_ranks)
        context_groups = create_groups_with_cp_ranks(config, ring_ranks)
        # LoongTrain further divides the ring group into subgroups
        window_size = config.lt_window_size
        if window_size == -1:
            window_size = ring_degree
        assert window_size <= ring_degree
        assert ring_degree % window_size == 0
        window_num = ring_degree // window_size
        interleaved = config.lt_interleaved
        if (window_size >= 8 or window_size == ring_degree) and interleaved:
            interleaved = False
            print_0(
                "WARNING: Disabling interleaved placement for large window size"
            )

        gpc._config = Config(
            {
                "selective_checkpoint": False,
                "parallel": {"sequence_2D": {"window_size": window_size}},
            }
        )
        gpc.is_forward = True

        intra_window_ranks = []
        inter_window_ranks = []
        if not interleaved:
            for i in range(head_degree):
                curr_ring_ranks = ring_ranks[i]
                for j in range(window_num):
                    intra_ranks = curr_ring_ranks[
                        j * window_size : (j + 1) * window_size
                    ]
                    intra_window_ranks.append(intra_ranks)
                for j in range(window_size):
                    inter_ranks = []
                    for t in range(window_num):
                        inter_ranks.append(
                            curr_ring_ranks[t * window_size + j]
                        )
                    inter_window_ranks.append(inter_ranks)
        # print_0(f"head_ranks: {head_ranks}")
        # print_0(f"ring_ranks: {ring_ranks}")
        # print_0(f"intra_window_ranks: {intra_window_ranks}")
        # print_0(f"inter_window_ranks: {inter_window_ranks}")
        intra_window_groups = create_groups_with_cp_ranks(
            config, intra_window_ranks
        )
        inter_window_groups = create_groups_with_cp_ranks(
            config, inter_window_ranks
        )
        dkv_intra_window_groups = create_groups_with_cp_ranks(
            config, intra_window_ranks
        )
        dkv_inter_window_groups = create_groups_with_cp_ranks(
            config, inter_window_ranks
        )

        local_head_group = None
        for group_id, ranks in enumerate(head_ranks):
            if dist.get_rank(group=config.cp_group) in ranks:
                local_head_group = head_groups[group_id]
                break
        assert local_head_group is not None
        local_context_group = None
        for group_id, ranks in enumerate(ring_ranks):
            if dist.get_rank(group=config.cp_group) in ranks:
                local_context_group = context_groups[group_id]
                break
        local_intra_window_group = None
        for group_id, ranks in enumerate(intra_window_ranks):
            if dist.get_rank(group=config.cp_group) in ranks:
                local_intra_window_group = intra_window_groups[group_id]
                break
        assert local_intra_window_group is not None
        local_inter_window_group = None
        for group_id, ranks in enumerate(inter_window_ranks):
            if dist.get_rank(group=config.cp_group) in ranks:
                local_inter_window_group = inter_window_groups[group_id]
                break
        assert local_inter_window_group is not None
        local_dkv_intra_window_group = None
        for group_id, ranks in enumerate(intra_window_ranks):
            if dist.get_rank(group=config.cp_group) in ranks:
                local_dkv_intra_window_group = dkv_intra_window_groups[
                    group_id
                ]
                break
        assert local_dkv_intra_window_group is not None
        local_dkv_inter_window_group = None
        for group_id, ranks in enumerate(inter_window_ranks):
            if dist.get_rank(group=config.cp_group) in ranks:
                local_dkv_inter_window_group = dkv_inter_window_groups[
                    group_id
                ]
                break
        assert local_dkv_inter_window_group is not None

        softmax_scale = q.shape[-1] ** (-0.5)
        lt_attn_mod = LTDistributedAttention(local_head_group)
        lt_attn_kwargs = {
            "dropout_p": config.dropout_p,
            "softmax_scale": softmax_scale,
            "causal": config.mask == "causal",
            "deterministic": config.deterministic,
            "context_group": local_context_group,
            "inter_window_group": local_inter_window_group,
            "intra_window_group": local_intra_window_group,
            "dkv_inter_window_group": local_dkv_inter_window_group,
            "dkv_intra_window_group": local_dkv_intra_window_group,
        }
        _LT_ATTN_MOD = lt_attn_mod
        _LT_ATTN_KWARGS = lt_attn_kwargs
        _local_context_group = local_context_group
        _local_head_group = local_head_group
    else:
        lt_attn_mod = _LT_ATTN_MOD
        lt_attn_kwargs = _LT_ATTN_KWARGS

    # loongtrain does not support padding mask or varlen
    # just pad and zero fill, as implemented in InternEvo
    cu_seqlens = F.pad(
        torch.cumsum(torch.tensor(seqlens, dtype=torch.int32), 0), (1, 0)
    ).to(torch.int32)
    assert torch.all(cu_seqlens % (2 * world_size) == 0)
    max_seqlen = max(seqlens)
    valuepadded_q = unpack_qkv_before_attn(q.unsqueeze(0), cu_seqlens)
    valuepadded_kv = unpack_qkv_before_attn(kv.unsqueeze(0), cu_seqlens)
    valuepadded_dout = unpack_qkv_before_attn(dout.unsqueeze(0), cu_seqlens)
    # LoongTrain does not implement reordering inside the attention module
    # so we need manual reorder input data
    assert (
        max_seqlen % (2 * world_size) == 0
    ), "Sequence length must be divisible by 2 * world_size"
    ring_rank = dist.get_rank(_local_context_group)
    ring_degree = dist.get_world_size(_local_context_group)
    head_rank = dist.get_rank(_local_head_group)
    head_degree = dist.get_world_size(_local_head_group)

    def _get_local_chunk(value: torch.Tensor):
        value_chunks = value.chunk(2 * ring_degree, dim=1)
        local_ring_group_value = torch.cat(
            [
                value_chunks[ring_rank],
                value_chunks[2 * ring_degree - 1 - ring_rank],
            ],
            dim=1,
        )
        # further slice based on head group
        local_val_chunks = local_ring_group_value.chunk(head_degree, dim=1)
        local_val = local_val_chunks[head_rank]
        return local_val

    def _zero_out_local_padding(value: torch.Tensor):
        mask_tensor = torch.full(
            (len(seqlens), max_seqlen), 1, dtype=torch.bool
        )
        for i, seqlen in enumerate(seqlens):
            mask_tensor[i, seqlen:] = 0
        mask_tensor = mask_tensor.to(value.device)
        local_mask = _get_local_chunk(mask_tensor)
        return value * local_mask.unsqueeze(-1).unsqueeze(-1)

    local_q = _get_local_chunk(valuepadded_q)
    local_kv = _get_local_chunk(valuepadded_kv)
    local_dout = _get_local_chunk(valuepadded_dout)
    return (
        local_q,
        local_kv,
        local_dout,
        lt_attn_mod,
        lt_attn_kwargs,
        _get_local_chunk,
        _zero_out_local_padding,
    )
