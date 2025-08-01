import os
from pathlib import Path
from typing import List, Callable, Optional

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import Dataset

from dcp.core.block_table import BlockType, WorkloadSpec
from dcp.core.common import ExecutionContext, ModelSpec
from dcp.core.cost_model import (
    AttnRooflineCostModel,
    CommunicationCostModel,
)
from dcp.core.instructions import BufferType, ExecutionPlan
from dcp.core.serialization import int_to_torch_dtype, torch_dtype_to_int
from dcp.data.dataloader import DCPDataLoader, TrainingSpec
from dcp.runtime.flash_attention import get_executor
from dcp.utils.logger import read_env_bool

DCP_SAVE_EXEC_PLAN_FOR_DEBUG = read_env_bool(
    "DCP_SAVE_EXEC_PLAN_FOR_DEBUG", default=False
)
DCP_LOAD_EXEC_PLAN_FOR_DEBUG = read_env_bool(
    "DCP_LOAD_EXEC_PLAN_FOR_DEBUG", default=False
)
assert not (DCP_SAVE_EXEC_PLAN_FOR_DEBUG and DCP_LOAD_EXEC_PLAN_FOR_DEBUG)


def _mask_to_tuple(attn_mask):
    if attn_mask is not None:
        attn_mask = attn_mask.cpu().tolist()
        if len(attn_mask) == 0:
            attn_mask = None
        else:
            if isinstance(attn_mask[0][0], int):
                # 2d mask
                attn_mask = tuple(tuple(x) for x in attn_mask)
            else:
                # 3d mask
                attn_mask = tuple(
                    tuple(tuple(x) for x in y) for y in attn_mask
                )
    return attn_mask


def get_byte_tensor(b: bytes):
    return torch.tensor(
        list(b), dtype=torch.uint8, device=torch.get_default_device()
    )


def read_byte_from_tensor(t: torch.Tensor):
    return bytes(t.cpu().tolist())


class DummyDataset(Dataset):
    def __init__(self, seqlens):
        self.size = len(seqlens)
        torch.manual_seed(42)
        # pre-generate all data
        self.seqlens = []
        self.data = []
        for seqlen in seqlens:
            self.seqlens.append(seqlen)
            result = {
                "text": torch.randint(0, 100, (seqlen,)).cpu(),
                "labels": torch.randint(0, 100, (seqlen,)).cpu(),
                "position_ids": torch.randint(0, 100, (seqlen,)).cpu(),
                "loss_mask": torch.ones((seqlen,), dtype=torch.float).cpu(),
            }
            self.data.append(result)

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        return self.data[index]


class DummyMultiModalDataset(Dataset):
    def __init__(self, text_lengths: List[List[int]], img_sizes: List[int]):
        assert len(text_lengths) == len(img_sizes)
        self.size = len(text_lengths)
        torch.manual_seed(42)
        # pre-generate all data
        self.text_lengths = text_lengths
        self.img_sizes = img_sizes
        self.data = []
        for lengths, img_size in zip(text_lengths, img_sizes):
            total_length = sum(lengths) + (len(lengths) - 1) * img_size
            result = {
                "text": torch.randint(0, 100, (total_length,)).cpu(),
                "labels": torch.randint(0, 100, (total_length,)).cpu(),
                "position_ids": torch.randint(0, 100, (total_length,)).cpu(),
                "loss_mask": torch.ones(
                    (total_length,), dtype=torch.float
                ).cpu(),
                "image_size": img_size,
                "text_lengths": lengths,
            }
            self.data.append(result)

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        return self.data[index]


class DummyBatchSampler(torch.utils.data.Sampler):
    def __init__(self, batch_indices: List[List[int]]):
        self.batch_indices = batch_indices
        self.size = len(batch_indices)

    def __iter__(self):
        return iter(self.batch_indices)

    def __len__(self):
        return self.size


def get_dataloader_for_benchmark(
    data_seqlens: List[int],
    batch_indices: List[List[int]],
    mask_type: str,
    n_heads: int,
    n_query_groups: int,
    head_dim: int,
    n_devices_per_node: int,
    block_size: int,
    head_block_size: int,
    qkv_dtype: torch.dtype,
    mem_imbalance_epsilon: float = 0.2,
    comp_imbalance_epsilon: float = 0.2,
    inter_node_comp_imbalance_factor: float = 5.0,
    num_preprocessing_workers=32,
    use_block_size_heuristic=False,
    mask_fn: Optional[Callable] = None,
    text_lengths: Optional[List[List[int]]] = None,
    img_sizes: Optional[List[int]] = None,
):
    n_total_devices = dist.get_world_size()
    n_nodes = n_total_devices // n_devices_per_node
    rank = dist.get_rank()
    node_rank = rank // n_devices_per_node
    local_rank = int(os.environ.get("LOCAL_RANK"))

    comm_cost_model = CommunicationCostModel()
    comp_cost_model = AttnRooflineCostModel()
    exec_context = ExecutionContext(
        n_devices_per_node,
        n_nodes,
        comm_cost_model=comm_cost_model,
        comp_cost_model=comp_cost_model,
    )

    train_spec = TrainingSpec(
        exec_context,
        ModelSpec(head_dim, n_heads, n_query_groups),
        mask_type,
        torch_dtype_to_int(torch.int32),
        torch_dtype_to_int(qkv_dtype),
        block_size,
        head_block_size,
        n_total_devices,
        use_block_size_heuristic=use_block_size_heuristic,
        mem_imbalance_epsilon=mem_imbalance_epsilon,
        comp_imbalance_epsilon=comp_imbalance_epsilon,
        inter_node_comp_imbalance_factor=inter_node_comp_imbalance_factor,
    )

    if text_lengths is not None:
        assert img_sizes is not None
        batched_dummy_dataset = DummyMultiModalDataset(text_lengths, img_sizes)
    else:
        batched_dummy_dataset = DummyDataset(data_seqlens)
    is_kv_host = rank == 0
    data_loader = DCPDataLoader(
        train_spec,
        batched_dummy_dataset,
        is_kv_host=is_kv_host,
        node_rank=node_rank,
        node_local_rank=local_rank,
        node_size=n_nodes,
        dcp_rank=rank,
        start_poller=local_rank == 0,
        batch_size=1,
        shuffle=False,
        batch_sampler=DummyBatchSampler(batch_indices),
        num_workers=1,
        num_preprocess_workers=num_preprocessing_workers,
        pin_memory=True,
        input_key="text",
        mask_fn=mask_fn,
        mask_fn_extra_keys=(
            ["image_size", "text_lengths"]
            if text_lengths is not None
            else None
        ),
    )
    return data_loader


def get_local_q_kv(
    q: torch.Tensor,
    kv: torch.Tensor,
    raw_seqlens: List[int],
    fw_exec_plan: ExecutionPlan,
    fw_workload: WorkloadSpec,
    node_rank: int,
    local_rank: int,
    head_block_size: int,
    bw_workload: Optional[WorkloadSpec] = None,
    dout: Optional[torch.Tensor] = None,
):
    local_cu_seqlens = fw_exec_plan.local_cu_seqlens
    local_q = torch.zeros(
        *fw_exec_plan.buffer_info[BufferType.LOCAL_Q].buffer_shape,
        dtype=q.dtype,
    )
    local_kv = torch.zeros(
        *fw_exec_plan.buffer_info[BufferType.LOCAL_KV].buffer_shape,
        dtype=kv.dtype,
    )
    global_cu_seqlens = np.cumsum([0] + raw_seqlens).tolist()
    for input_id in range(len(fw_workload.input_sizes)):
        if fw_workload.input_to_device_map[input_id] == (
            node_rank,
            local_rank,
        ):
            buffer_id = fw_workload.block_mapping.input_id_to_buffer_index[
                (node_rank, local_rank)
            ][input_id]
            block_meta = fw_workload.block_mapping.input_id_to_meta[input_id]
            local_token_offset = local_cu_seqlens[buffer_id]
            head_offset = block_meta.head_id * block_meta.head_block_size

            global_token_offset = (
                global_cu_seqlens[block_meta.seq_id]
                + block_meta.block_id * block_meta.block_size
            )
            if block_meta.type == BlockType.Q:
                local_q[
                    local_token_offset : local_token_offset
                    + block_meta.n_tokens,
                ] = q[
                    global_token_offset : global_token_offset
                    + block_meta.n_tokens,
                    head_offset : head_offset + block_meta.head_block_size,
                ]
            elif block_meta.type == BlockType.KV:
                local_kv[
                    local_token_offset : local_token_offset
                    + block_meta.n_tokens,
                ] = kv[
                    global_token_offset : global_token_offset
                    + block_meta.n_tokens,
                    :,
                    head_offset : head_offset + block_meta.head_block_size,
                ]
    if dout is not None:
        local_dout = torch.zeros(
            local_cu_seqlens[-1],
            head_block_size,
            q.shape[-1],
            dtype=dout.dtype,
        )
        for dout_id in bw_workload.block_mapping.input_id_to_buffer_index[
            (node_rank, local_rank)
        ]:
            buffer_id = bw_workload.block_mapping.input_id_to_buffer_index[
                (node_rank, local_rank)
            ][dout_id]
            block_meta = bw_workload.block_mapping.input_id_to_meta[dout_id]
            local_token_offset = local_cu_seqlens[buffer_id]
            head_offset = block_meta.head_id * block_meta.head_block_size
            global_token_offset = (
                global_cu_seqlens[block_meta.seq_id]
                + block_meta.block_id * block_meta.block_size
            )
            if block_meta.type == BlockType.dOut:
                local_dout[
                    local_token_offset : local_token_offset
                    + block_meta.n_tokens,
                ] = dout[
                    global_token_offset : global_token_offset
                    + block_meta.n_tokens,
                    head_offset : head_offset + block_meta.head_block_size,
                ]
    else:
        local_dout = None
    return local_q, local_kv, local_dout


def prepare_dcp_distributed_attn_for_test(
    q: torch.Tensor,  # [sum(raw_seqlens), n_heads, head_dim]
    kv: torch.Tensor,  # [sum(raw_seqlens), 2, n_query_groups, head_dim]
    mask_type: str,
    raw_seqlens,
    n_devices_per_node,
    block_size,
    head_block_size,
    comm_cost_model=None,
    comp_cost_model=None,
    dout=None,
    mem_imbalance_epsilon=0.2,
    comp_imbalance_epsilon=0.2,
    inter_node_comp_imbalance_factor=5.0,
    dropout_p=0.0,
    window_size=(-1, -1),
    reduction_schedule_algo="delayed",
    executor_impl="python",
    deterministic=False,
    synchronous=False,
    mask_fn: Optional[Callable] = None,
    use_block_size_heuristic=False,
    text_lengths: Optional[List[List[int]]] = None,
    img_sizes: Optional[List[int]] = None,
):
    n_total_devices = dist.get_world_size()
    n_nodes = n_total_devices // n_devices_per_node
    rank = dist.get_rank()
    node_rank = rank // n_devices_per_node
    local_rank = int(os.environ.get("LOCAL_RANK"))

    d = q.shape[-1]
    assert d == kv.shape[-1]
    assert q.dtype == kv.dtype
    n_query_groups = kv.shape[-2]
    n_heads = q.shape[-2]

    softmax_scale = d ** (-0.5)

    if comm_cost_model is None:
        comm_cost_model = CommunicationCostModel()
    if comp_cost_model is None:
        comp_cost_model = AttnRooflineCostModel()
    exec_context = ExecutionContext(
        n_devices_per_node,
        n_nodes,
        comm_cost_model=comm_cost_model,
        comp_cost_model=comp_cost_model,
    )

    train_spec = TrainingSpec(
        exec_context,
        ModelSpec(d, n_heads, n_query_groups),
        mask_type,
        torch_dtype_to_int(torch.int32),
        torch_dtype_to_int(q.dtype),
        block_size,
        head_block_size,
        n_total_devices,
        use_block_size_heuristic=use_block_size_heuristic,
        mem_imbalance_epsilon=mem_imbalance_epsilon,
        comp_imbalance_epsilon=comp_imbalance_epsilon,
        inter_node_comp_imbalance_factor=inter_node_comp_imbalance_factor,
    )
    if text_lengths is not None:
        assert img_sizes is not None
        dummy_dataset = DummyMultiModalDataset(text_lengths, img_sizes)
    else:
        dummy_dataset = DummyDataset(raw_seqlens)
    is_kv_host = rank == 0
    data_loader = DCPDataLoader(
        train_spec,
        dummy_dataset,
        is_kv_host=is_kv_host,
        node_rank=node_rank,
        node_local_rank=local_rank,
        node_size=n_nodes,
        dcp_rank=rank,
        start_poller=local_rank == 0,
        batch_size=len(raw_seqlens),
        shuffle=False,
        num_workers=1,
        num_preprocess_workers=1,
        pin_memory=True,
        input_key="text",
        mask_fn=mask_fn,
        mask_fn_extra_keys=(
            ["image_size", "text_lengths"]
            if text_lengths is not None
            else None
        ),
    )

    (
        _,
        fw_exec_plan,
        bw_exec_plan,
        fw_workload,
        bw_workload,
        _,
    ) = next(iter(data_loader))
    fw_exec_plan: ExecutionPlan
    bw_exec_plan: ExecutionPlan
    fw_workload: WorkloadSpec
    bw_workload: WorkloadSpec

    executor = get_executor(
        head_block_size,
        q.shape[-1],
        exec_context,
        executor_impl=executor_impl,
        synchronous=synchronous,
    )

    local_q, local_kv, local_dout = get_local_q_kv(
        q,
        kv,
        raw_seqlens,
        fw_exec_plan,
        fw_workload,
        node_rank,
        local_rank,
        head_block_size,
        bw_workload=bw_workload,
        dout=dout,
    )
    return (
        data_loader,
        executor,
        fw_exec_plan,
        bw_exec_plan,
        local_q,
        local_kv,
        local_dout,
        fw_workload,
        bw_workload,
    )
