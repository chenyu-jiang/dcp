# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import logging
import os
import time
import traceback
import math
from dataclasses import dataclass, fields
from functools import partial
from itertools import product
from queue import Empty
from typing import Dict, List, Optional, Tuple, Callable

import pickle

import numpy as np
import torch

# import multiprocessing as mp
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.utils.data import DataLoader as _DataLoader

from dcp.core.block_table import (
    BlockMappings,
    BlockMeta,
    BlockType,
    ComputationBlockMeta,
    DataBlockMeta,
    WorkloadSpec,
)
from dcp.core.common import ExecutionContext, ModelSpec
from dcp.core.compiler import CompilerConfig, InstrCompiler
from dcp.core.cost_model import (
    AttnRooflineCostModel,
    CommunicationCostModel,
)
from dcp.core.instructions import BufferType, DType, ExecutionPlan
from dcp.core.serialization import int_to_torch_dtype, torch_dtype_to_int
from dcp.data.kv_redis import RedisKVStore, RedisServer
from dcp.utils.common import trepr
from dcp.utils.logger import create_logger, logger, read_env_bool

DCP_DATALOADER_LOG_KV = read_env_bool("DCP_DATALOADER_LOG_KV", False)

MANAGER_PROCESS_TIMEOUT = 1
RECEIVER_PROCESS_TIMEOUT = 1
KVSTORE_TIMEOUT = 1800  # 30 minutes

# ONLY USED FOR DEBUG PURPOSES
DEBUG_USE_DUMMY_EP = False
DEBUG_DUMP_EP_STATS = os.getenv(
    "DCP_DEBUG_DUMP_EP_STATS", "False"
).lower() in ("true", "1", "t")
DEBUG_DUMP_EP_PREFIX = os.environ.get("DCP_DEBUG_DUMP_EP_PREFIX", None)
if DEBUG_DUMP_EP_STATS and DEBUG_DUMP_EP_PREFIX is None:
    raise ValueError(
        "DCP_DEBUG_DUMP_EP_PREFIX must be set if "
        "DCP_DEBUG_DUMP_EP_STATS is set."
    )

_kvstore_handle = None


def _get_kv_host_port():
    host = os.environ.get("DCP_KV_HOST", "localhost")
    port = os.environ.get("DCP_KV_PORT", 29500)
    return host, port


def _init_kv_server(logger=None):
    host, port = _get_kv_host_port()
    if logger is not None:
        logger.debug("Init kv server, host: {}, port: {}".format(host, port))
    kv_server = RedisServer(host, port)
    return kv_server, host, port


def _init_kv_store(logger=None):
    host, port = _get_kv_host_port()
    if logger is not None:
        logger.debug("Init kv store, host: {}, port: {}".format(host, port))
    kv_store = RedisKVStore(host, port)
    return kv_store, host, port


def _checked_delete_key(kv_store: RedisKVStore, key: str, logger=None):
    result = kv_store.delete_key(key)
    if not result:
        raise RuntimeError(
            "Internal error: failed to delete key " "{}.".format(key)
        )
    if logger is not None and DCP_DATALOADER_LOG_KV:
        logger.debug("Deleted key: {}".format(key))


def _get_from_shared_kv_store(
    kv_store: RedisKVStore,
    key: str,
    reader_idx: int,
    n_total_readers: int,
    decode: bool = True,
    logger=None,
):
    reader_count_key = key + "_rc"
    reader_ack_key = key + "_r{}_ack".format(reader_idx)
    # wait for reader ack
    if logger is not None and DCP_DATALOADER_LOG_KV:
        logger.debug("Waiting for reader ack key: {}".format(reader_ack_key))
    kv_store.get(reader_ack_key)
    if logger is not None and DCP_DATALOADER_LOG_KV:
        logger.debug(
            "Got reader ack key: {}, waiting for data key: {}".format(
                reader_ack_key, key
            )
        )
    data = kv_store.get(key)
    if logger is not None and DCP_DATALOADER_LOG_KV:
        logger.debug("Removing reader ack key: {}".format(reader_ack_key))
    # remove reader ack
    _checked_delete_key(kv_store, reader_ack_key, logger=logger)
    # get reader count
    reader_count = kv_store.add(reader_count_key, 1)
    if reader_count == n_total_readers:
        if logger is not None and DCP_DATALOADER_LOG_KV:
            logger.debug(
                "Last reader, reset reader count: {}".format(reader_count_key)
            )
        # reset reader count
        result_readers = kv_store.add(reader_count_key, -n_total_readers)
        assert result_readers == 0
        if logger is not None and DCP_DATALOADER_LOG_KV:
            logger.debug("Last reader, remove data key: {}".format(key))
        # remove data key
        _checked_delete_key(kv_store, key, logger=logger)
        if logger is not None and DCP_DATALOADER_LOG_KV:
            logger.debug("Last reader, set ack key: {}".format(key + "_ack"))
        # set all reader ack keys
        keys_to_reset = [
            key + "_r{}_ack".format(i) for i in range(n_total_readers)
        ]
        if logger is not None and DCP_DATALOADER_LOG_KV:
            logger.debug("Last reader, reset keys: {}".format(keys_to_reset))
        for reset_key in keys_to_reset:
            val = kv_store.add(reset_key, 1)
            # make sure the key is set
            got_val = int(kv_store.get(reset_key).decode())
            if not val == got_val:
                raise RuntimeError(
                    "Failed to set reader ack key: {}".format(reset_key)
                )
            if logger is not None and DCP_DATALOADER_LOG_KV:
                logger.debug("Set reader ack key: {}".format(reset_key))
        # set data ack key
        kv_store.add(key + "_ack", 1)

    if decode:
        return data.decode()
    return data


def _put_to_shared_kv_store(
    kv_store: RedisKVStore, key: str, data, logger=None
):
    # put execution plan into local kv store
    ack_key = key + "_ack"
    if logger is not None and DCP_DATALOADER_LOG_KV:
        logger.debug("Wait for data ack key: {}".format(ack_key))
    # wait for ack key
    kv_store.get(ack_key)
    # remove ack key
    _checked_delete_key(kv_store, ack_key, logger=logger)
    if logger is not None and DCP_DATALOADER_LOG_KV:
        logger.debug("Set data key: {}".format(key))
    # set data key
    kv_store.set(key, data)


@dataclass
class WorkerData:
    logger: Optional[logging.Logger] = None
    kv_store: Optional[RedisKVStore] = None
    processed_batches: Optional[int] = None
    kv_buffer_size: Optional[int] = None

    def check_initialized(self):
        cls_fields = fields(self.__class__)
        for fld in cls_fields:
            if fld.name == "mask_fn":
                continue
            if getattr(self, fld.name) is None:
                raise RuntimeError(
                    "Worker data not initialized: {}".format(fld.name)
                )


@dataclass
class PlannerWorkerData(WorkerData):
    # required at initialization:
    node_rank: Optional[int] = None
    dcp_size: Optional[int] = None
    mask_fn: Optional[Callable] = None
    is_dryrun: Optional[bool] = False
    # filled later in worker init:
    exec_planner: Optional["ExecutionPlanner"] = None
    assigned_iters_per_node: Optional[int] = None
    node_size: Optional[int] = None

    def __post_init__(self):
        if self.node_rank is None:
            raise RuntimeError("node_rank must be set at initialization.")


@dataclass
class DataloaderWorkerData(WorkerData):
    # required at initialization:
    dcp_rank: Optional[int] = None
    pp_rank: Optional[int] = None
    tp_rank: Optional[int] = None
    is_dryrun: Optional[bool] = False
    # filled later in worker init:
    dcp_size: Optional[int] = None
    # pp_size: Optional[int] = None
    tp_size: Optional[int] = None
    block_size: Optional[int] = None

    def __post_init__(self):
        if self.dcp_rank is None:
            raise RuntimeError("dcp_rank must be set at initialization.")
        if self.pp_rank is None:
            raise RuntimeError("pp_rank must be set at initialization.")
        if self.tp_rank is None:
            raise RuntimeError("tp_rank must be set at initialization.")


class KVStoreMetaKeys:
    DCP_SIZE = "data_context_parallel_size"
    TP_SIZE = "tensor_parallel_size"
    PP_SIZE = "pipeline_parallel_size"
    EXEC_CONTEXT = "exec_context"
    MODEL_SPEC = "model_spec"
    MASK_TYPE = "mask_type"
    MASK_DTYPE = "mask_dtype"
    QKV_DTYPE = "qkv_dtype"
    BLOCK_SIZE = "block_size"
    HEAD_BLOCK_SIZE = "head_block_size"
    KV_BUFFER_SIZE = "kv_buffer_size"
    ASSIGNED_ITER_PER_NODE = "assigned_iters_per_node"
    USE_BLOCK_SIZE_HEURISTIC = "use_block_size_heuristic"
    MEM_IMBALANCE_EPSILON = "mem_imbalance_epsilon"
    COMP_IMBALANCE_EPSILON = "comp_imbalance_epsilon"
    INTER_NODE_COMP_IMBALANCE_FACTOR = "inter_node_comp_imbalance_factor"
    # used outside dataloader
    N_ITERS = "n_iters"


@dataclass
class TrainingSpec:
    exec_context: ExecutionContext
    model_spec: ModelSpec
    mask_type: str
    mask_dtype: int
    qkv_dtype: int
    block_size: int
    head_block_size: int
    dcp_size: int
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    n_microbatches: int = 1
    use_block_size_heuristic: int = 1
    mem_imbalance_epsilon: float = 0.2
    comp_imbalance_epsilon: float = 0.2
    inter_node_comp_imbalance_factor: float = 5.0

    def __post_init__(self):
        if isinstance(self.use_block_size_heuristic, bool):
            self.use_block_size_heuristic = int(self.use_block_size_heuristic)

    def __str__(self):
        return (
            f"TrainingSpec("
            f"exec_context={self.exec_context}, "
            f"model_spec={self.model_spec}, "
            f"mask_type={self.mask_type}, "
            f"mask_dtype={self.mask_dtype}, "
            f"qkv_dtype={self.qkv_dtype}, "
            f"block_size={self.block_size}, "
            f"head_block_size={self.head_block_size}, "
            f"tensor_parallel_size={self.tensor_parallel_size}, "
            f"pipeline_parallel_size={self.pipeline_parallel_size}, "
            f"n_microbatches={self.n_microbatches}, "
            f"dcp_size={self.dcp_size}, "
            f"use_block_size_heuristic={self.use_block_size_heuristic},"
            f"mem_imbalance_epsilon={self.mem_imbalance_epsilon},"
            f"comp_imbalance_epsilon={self.comp_imbalance_epsilon},"
            f"inter_node_comp_imbalance_factor={self.inter_node_comp_imbalance_factor}"
        )


def _preprocessing_worker_init_fn(worker_id):
    torch.set_default_device("cpu")
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset

    worker_data: PlannerWorkerData = dataset.worker_data
    # local information such as rank should be stored in the dataset
    node_rank = worker_data.node_rank

    logger = create_logger(
        "preprocess_worker",
        prefix=f"Node {node_rank} | " f"Preprocessing Worker {worker_id}",
        log_file="preprocessing/" f"nr{node_rank}_w{worker_id}.log",
    )
    # init kv store first since we need to get data from it
    kv_store, host, port = _init_kv_store(logger=logger)
    assigned_iters_per_node = int(
        kv_store.get(KVStoreMetaKeys.ASSIGNED_ITER_PER_NODE).decode()
    )
    worker_data.assigned_iters_per_node = assigned_iters_per_node

    worker_data.logger = logger
    worker_data.logger.debug("Subprocess started.")
    worker_data.kv_store = kv_store
    worker_data.processed_batches = 0

    # create data opt
    model_spec, _ = ModelSpec.deserialize(
        kv_store.get(KVStoreMetaKeys.MODEL_SPEC).decode().encode("iso-8859-1")
    )
    exec_context, _ = ExecutionContext.deserialize(
        kv_store.get(KVStoreMetaKeys.EXEC_CONTEXT)
        .decode()
        .encode("iso-8859-1")
    )
    mask_type = kv_store.get(KVStoreMetaKeys.MASK_TYPE).decode()
    mask_dtype = int(kv_store.get(KVStoreMetaKeys.MASK_DTYPE).decode())
    qkv_dtype = int(kv_store.get(KVStoreMetaKeys.QKV_DTYPE).decode())
    block_size = int(kv_store.get(KVStoreMetaKeys.BLOCK_SIZE).decode())
    head_block_size = int(
        kv_store.get(KVStoreMetaKeys.HEAD_BLOCK_SIZE).decode()
    )
    dcp_size = int(kv_store.get(KVStoreMetaKeys.DCP_SIZE).decode())
    use_block_size_heuristic = int(
        kv_store.get(KVStoreMetaKeys.USE_BLOCK_SIZE_HEURISTIC).decode()
    )
    mem_imbalance_epsilon = float(
        kv_store.get(KVStoreMetaKeys.MEM_IMBALANCE_EPSILON).decode()
    )
    comp_imbalance_epsilon = float(
        kv_store.get(KVStoreMetaKeys.COMP_IMBALANCE_EPSILON).decode()
    )
    inter_node_comp_imbalance_factor = float(
        kv_store.get(KVStoreMetaKeys.INTER_NODE_COMP_IMBALANCE_FACTOR).decode()
    )

    worker_data.dcp_size = dcp_size
    # create exec planner
    mask_fn = worker_data.mask_fn
    exec_planner = ExecutionPlanner(
        node_rank,
        worker_id,
        exec_context,
        model_spec,
        mask_type,
        mask_dtype,
        qkv_dtype,
        block_size,
        head_block_size,
        use_block_size_heuristic=use_block_size_heuristic,
        mem_epsilon=mem_imbalance_epsilon,
        comp_epsilon=comp_imbalance_epsilon,
        inter_node_comp_imbalance_factor=inter_node_comp_imbalance_factor,
        mask_fn=mask_fn,
    )
    worker_data.exec_planner = exec_planner
    worker_data.node_size = exec_context.n_nodes

    kv_buffer_size = int(kv_store.get(KVStoreMetaKeys.KV_BUFFER_SIZE).decode())
    worker_data.kv_buffer_size = kv_buffer_size
    worker_data.check_initialized()
    worker_data.logger.debug("Exiting init function.")


def _worker_init_fn(worker_id):
    torch.set_default_device("cpu")
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset

    worker_data: DataloaderWorkerData = dataset.worker_data
    # local information such as rank should be stored in the dataset
    dcp_rank = worker_data.dcp_rank
    tp_rank = worker_data.tp_rank

    logger = create_logger(
        "worker",
        prefix=f"dcRank {dcp_rank} tRank {tp_rank}| "
        f"Dataloader Worker {worker_id}",
        log_file="dataloader/" f"dcr{dcp_rank}t{tp_rank}_w{worker_id}.log",
    )
    # init kv store first since we need to get data from it
    kv_store, host, port = _init_kv_store(logger=logger)
    dcp_size = int(kv_store.get(KVStoreMetaKeys.DCP_SIZE).decode())
    tp_size = int(kv_store.get(KVStoreMetaKeys.TP_SIZE).decode())
    block_size = int(kv_store.get(KVStoreMetaKeys.BLOCK_SIZE).decode())
    worker_data.dcp_size = dcp_size
    worker_data.tp_size = tp_size
    worker_data.logger = logger
    worker_data.logger.debug("Subprocess started.")
    worker_data.kv_store = kv_store
    worker_data.processed_batches = 0
    kv_buffer_size = int(kv_store.get(KVStoreMetaKeys.KV_BUFFER_SIZE).decode())
    worker_data.kv_buffer_size = kv_buffer_size
    worker_data.block_size = block_size
    worker_data.check_initialized()
    worker_data.logger.debug("Exiting init function.")


# here collate samples copies input batch into buffer locations
# according to the execution plan received indices
def _collate_samples(
    batch,
    padded_seqlens: List[List[int]],
    fw_workload_spec: WorkloadSpec,
    input_key: str = "text",
    node_rank: int = 0,
    node_local_rank: int = 0,
    logger: Optional[logging.Logger] = None,
):
    # get block size from fw_workload_spec
    block_size = fw_workload_spec.block_mapping.input_id_to_meta[0].block_size
    # split input sequences into blocks
    input_blocks = []
    label_blocks = []
    input_positions = []
    input_loss_masks = []
    for seq_id, seq_dict in enumerate(batch):
        seq = seq_dict[input_key]
        label = seq_dict["labels"]
        position_ids = seq_dict["position_ids"]
        loss_mask = seq_dict["loss_mask"]
        input_blocks.extend(
            [seq[i : i + block_size] for i in range(0, len(seq), block_size)]
        )
        label_blocks.extend(
            [
                label[i : i + block_size]
                for i in range(0, len(label), block_size)
            ]
        )
        input_positions.extend(
            [
                position_ids[i : i + block_size]
                for i in range(0, len(position_ids), block_size)
            ]
        )
        input_loss_masks.extend(
            [
                loss_mask[i : i + block_size]
                for i in range(0, len(loss_mask), block_size)
            ]
        )
        logger.debug(
            "SeqId: {}, N Input blocks: {}.".format(
                seq_id,
                len(list(range(0, len(seq), block_size))),
            )
        )

    def _seq_block(input_id):
        block_meta = fw_workload_spec.block_mapping.input_id_to_meta[input_id]
        return (block_meta.seq_id, block_meta.block_id)

    cumsum_padded_seqlens = np.cumsum([0] + padded_seqlens)
    block_ids = set()
    for input_id in sorted(
        range(len(fw_workload_spec.input_sizes)), key=_seq_block
    ):
        if fw_workload_spec.input_to_device_map[input_id] == (
            node_rank,
            node_local_rank,
        ):
            block_meta = fw_workload_spec.block_mapping.input_id_to_meta[
                input_id
            ]
            if block_meta.type in [BlockType.Q, BlockType.KV]:
                block_id = (
                    cumsum_padded_seqlens[block_meta.seq_id] // block_size
                    + block_meta.block_id
                )
                block_ids.add(block_id)

    block_ids = sorted(list(block_ids))
    input_blocks = [input_blocks[i] for i in block_ids]
    label_blocks = [label_blocks[i] for i in block_ids]
    input_positions = [input_positions[i] for i in block_ids]
    input_loss_masks = [input_loss_masks[i] for i in block_ids]

    input_blocks = torch.cat(input_blocks)
    label_blocks = torch.cat(label_blocks)
    input_positions = torch.cat(input_positions)
    input_loss_masks = torch.cat(input_loss_masks)
    input_blocks = input_blocks.unsqueeze(0)
    label_blocks = label_blocks.unsqueeze(0)
    input_positions = input_positions.unsqueeze(0)
    input_loss_masks = input_loss_masks.unsqueeze(0)

    batch_ = {
        input_key: input_blocks,
        "labels": label_blocks,
        "position_ids": input_positions,
        "loss_mask": input_loss_masks,
    }

    return batch_


def _preprocessing_collate_fn(
    batch,
    input_key="text",
    mask_fn_extra_keys: Optional[List[str]] = None,
):
    # get states from variables set in worker_init_fn
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset
    worker_data: PlannerWorkerData = dataset.worker_data
    try:
        # needed by both planner and worker
        kv_store = worker_data.kv_store
        kv_buffer_size = worker_data.kv_buffer_size
        # processed_batches is local to each worker since dataset
        # is replicated
        processed_batches = worker_data.processed_batches

        assigned_iters_per_node = worker_data.assigned_iters_per_node
        # pytorch assigns batch to workers in round-robin fashion
        # so we can use worker_info.id to determine the current batch index
        # TODO: find a better way to do this since it is not documented
        #       and depends on implementation
        current_batch_idx = (
            worker_info.num_workers * processed_batches + worker_info.id
        )
        assigned_node_id = (current_batch_idx // assigned_iters_per_node) % (
            worker_data.node_size
        )
        worker_data.logger.debug(
            "Iteration: {}, assigned_node_id: {}".format(
                current_batch_idx, assigned_node_id
            )
        )
        # increase processed_batches
        worker_info.dataset.worker_data.processed_batches += 1
        if (
            assigned_node_id != worker_data.node_rank
            and not worker_data.is_dryrun
        ):
            # directly return
            worker_data.logger.debug(
                f"Skipped generating EP for iteration {current_batch_idx}."
            )
            return None

        buffer_slot = current_batch_idx % kv_buffer_size
        ep_key = f"ep_{buffer_slot}"
        wl_key = f"wl_{buffer_slot}"
        sl_key = f"sl_{buffer_slot}"

        exec_planner: ExecutionPlanner = worker_data.exec_planner
        # calculate exec plans on planner
        input_seqlens = []
        mask_fn_info = []
        for sample in batch:
            if isinstance(sample, list):
                assert mask_fn_extra_keys is None
                seq = sample
            elif isinstance(sample, dict):
                assert input_key in sample, (
                    f"Input key '{input_key}' not found in sample. "
                    f"Available keys: {sample.keys()}"
                )
                if mask_fn_extra_keys is not None:
                    mask_fn_info_dict = {}
                    for k in mask_fn_extra_keys:
                        assert k in sample, (
                            f"Mask extra key '{k}' not found in sample. "
                            f"Available keys: {sample.keys()}"
                        )
                        mask_fn_info_dict[k] = sample[k]
                    mask_fn_info.append(mask_fn_info_dict)
                seq = sample[input_key]
            else:
                raise ValueError(
                    "sample must be either a list or a dict"
                    " but got {}".format(type(sample))
                )
            seqlen = len(seq)
            input_seqlens.append(seqlen)

        worker_data.logger.debug(
            f"Generating EP for iteration {current_batch_idx}."
        )

        # Utilize these seq lens to get the execution plan.

        t_start = time.time()
        t_gen_sch_start = time.time()
        (
            fw_exec_plans,
            bw_exec_plans,
            fw_workload_spec,
            bw_workload_spec,
            padded_seqlens,
        ) = exec_planner.plan(
            input_seqlens,
            iteration_idx=current_batch_idx,
            mask_fn_info=mask_fn_info,
        )
        t_gen_sch_end = time.time()
        worker_data.logger.debug(
            "Schedule generation for iteration {} took {} seconds.".format(
                current_batch_idx,
                t_gen_sch_end - t_gen_sch_start,
            )
        )

        worker_data.logger.debug(
            "Finished generating execution plan for iteration "
            f"{current_batch_idx}, "
        )
        t_end = time.time()
        worker_data.logger.debug(
            "EP generation for iteration {} took {} seconds.".format(
                current_batch_idx, t_end - t_start
            )
        )
        for d in fw_exec_plans.keys():
            dcp_rank_ = (
                d[0] * exec_planner.exec_context.n_devices_per_node + d[1]
            )
            fw_ep = fw_exec_plans[d]
            if bw_exec_plans is not None:
                bw_ep = bw_exec_plans[d]
            else:
                bw_ep = None
            t_ser_start = time.time()
            # serialized_fw_ep = fw_ep.serialize().decode("iso-8859-1")
            serialized_fw_ep = pickle.dumps(fw_ep).decode("iso-8859-1")
            t_ser_end = time.time()
            worker_data.logger.debug(
                "Serialized fw EP for rank {} in {} seconds.".format(
                    dcp_rank_, t_ser_end - t_ser_start
                )
            )
            t_ser_start = time.time()
            # serialized_bw_ep = bw_ep.serialize().decode("iso-8859-1")
            serialized_bw_ep = pickle.dumps(bw_ep).decode("iso-8859-1")
            t_ser_end = time.time()
            worker_data.logger.debug(
                "Serialized bw EP for rank {} in {} seconds.".format(
                    dcp_rank_, t_ser_end - t_ser_start
                )
            )
            t_ser_start = time.time()
            serialized_fw_workload_spec = (
                # fw_workload_spec.serialize().decode("iso-8859-1")
                pickle.dumps(fw_workload_spec).decode("iso-8859-1")
            )
            t_ser_end = time.time()
            worker_data.logger.debug(
                "Serialized fw workload spec for rank {} in {} seconds.".format(
                    dcp_rank_, t_ser_end - t_ser_start
                )
            )
            t_ser_start = time.time()
            serialized_bw_workload_spec = (
                # bw_workload_spec.serialize().decode("iso-8859-1")
                pickle.dumps(bw_workload_spec).decode("iso-8859-1")
            )
            t_ser_end = time.time()
            worker_data.logger.debug(
                "Serialized bw workload spec for rank {} in {} seconds.".format(
                    dcp_rank_, t_ser_end - t_ser_start
                )
            )
            serialized_seqlens = json.dumps(padded_seqlens)
            worker_data.logger.debug(
                f"Pushing DCP rank {dcp_rank_} Device {d}'s "
                f"execution plan for iter {current_batch_idx} to kv store."
            )
            if worker_data.is_dryrun and dcp_rank_ != 0:
                continue
            # put execution plan into kv store
            _put_to_shared_kv_store(
                kv_store,
                f"r{dcp_rank_}_fw_" + ep_key,
                serialized_fw_ep,
                logger=worker_data.logger,
            )
            _put_to_shared_kv_store(
                kv_store,
                f"r{dcp_rank_}_bw_" + ep_key,
                serialized_bw_ep,
                logger=worker_data.logger,
            )
            _put_to_shared_kv_store(
                kv_store,
                f"r{dcp_rank_}_fw_" + wl_key,
                serialized_fw_workload_spec,
                logger=worker_data.logger,
            )
            _put_to_shared_kv_store(
                kv_store,
                f"r{dcp_rank_}_bw_" + wl_key,
                serialized_bw_workload_spec,
                logger=worker_data.logger,
            )
            _put_to_shared_kv_store(
                kv_store,
                f"r{dcp_rank_}_" + sl_key,
                serialized_seqlens,
                logger=worker_data.logger,
            )
        worker_data.logger.debug(
            "Successfully pushed EP "
            f"{current_batch_idx} to shared kv store."
        )
        t_end = time.time()
        worker_data.logger.debug(
            "EP generation for iteration {} took {} seconds.".format(
                current_batch_idx, t_end - t_start
            )
        )
    except Exception as e:
        # explicitly log exception here since it will be swallowed by
        # multiprocessing
        worker_data.logger.error("Exception in worker process: {}".format(e))
        worker_data.logger.error(traceback.format_exc())
        raise e
    return None


def get_preprocessing_collate_fn(
    input_key="text",
    mask_fn_extra_keys: Optional[List[str]] = None,
):
    return partial(
        _preprocessing_collate_fn,
        input_key=input_key,
        mask_fn_extra_keys=mask_fn_extra_keys,
    )


def _collate_fn(
    batch,
    input_key: str = "text",
    training_spec: TrainingSpec = None,
    node_rank: int = 0,
    node_local_rank: int = 0,
):
    # get states from variables set in worker_init_fn
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset
    worker_data: DataloaderWorkerData = dataset.worker_data
    try:
        dcp_rank = worker_data.dcp_rank
        kv_store = worker_data.kv_store
        kv_buffer_size = worker_data.kv_buffer_size
        # processed_batches is local to each worker since dataset
        # is replicated
        processed_batches = worker_data.processed_batches

        # pytorch assigns batch to workers in round-robin fashion
        # so we can use worker_info.id to determine the current batch index
        # TODO: find a better way to do this since it is not documented
        #       and depends on implemtation
        current_batch_idx = (
            worker_info.num_workers * processed_batches + worker_info.id
        )
        buffer_slot = current_batch_idx % kv_buffer_size
        ep_key = f"ep_{buffer_slot}"
        wl_key = f"wl_{buffer_slot}"
        sl_key = f"sl_{buffer_slot}"

        worker_data.logger.debug(
            "Waiting for EP for iteration {}.".format(current_batch_idx)
        )
        n_readers = worker_data.tp_size if not worker_data.is_dryrun else 1
        reader_idx = worker_data.tp_rank
        worker_data.logger.debug(
            "N readers: {}, reader_id: {}".format(n_readers, reader_idx)
        )
        serialized_fw_ep = (
            _get_from_shared_kv_store(
                kv_store,
                f"r{dcp_rank}_fw_" + ep_key,
                reader_idx=reader_idx,
                n_total_readers=n_readers,
                decode=False,
                logger=worker_data.logger,
            )
            .decode()
            .encode("iso-8859-1")
        )
        serialized_bw_ep = (
            _get_from_shared_kv_store(
                kv_store,
                f"r{dcp_rank}_bw_" + ep_key,
                reader_idx=reader_idx,
                n_total_readers=n_readers,
                decode=False,
                logger=worker_data.logger,
            )
            .decode()
            .encode("iso-8859-1")
        )
        serialized_fw_workload_spec = (
            _get_from_shared_kv_store(
                kv_store,
                f"r{dcp_rank}_fw_" + wl_key,
                reader_idx=reader_idx,
                n_total_readers=n_readers,
                decode=False,
                logger=worker_data.logger,
            )
            .decode()
            .encode("iso-8859-1")
        )
        serialized_bw_workload_spec = (
            _get_from_shared_kv_store(
                kv_store,
                f"r{dcp_rank}_bw_" + wl_key,
                reader_idx=reader_idx,
                n_total_readers=n_readers,
                decode=False,
                logger=worker_data.logger,
            )
            .decode()
            .encode("iso-8859-1")
        )
        padded_seqlens = json.loads(
            _get_from_shared_kv_store(
                kv_store,
                f"r{dcp_rank}_" + sl_key,
                reader_idx=reader_idx,
                n_total_readers=n_readers,
                decode=False,
                logger=worker_data.logger,
            )
        )
        worker_data.logger.debug(
            "Got data for iteration {}.".format(current_batch_idx)
        )
        t_deser_start = time.time()
        # fw_ep, _ = ExecutionPlan.deserialize(serialized_fw_ep)
        fw_ep = pickle.loads(serialized_fw_ep)
        t_deser_end = time.time()
        worker_data.logger.debug(
            "Deserialized fw EP for iteration {} in {} seconds.".format(
                current_batch_idx, t_deser_end - t_deser_start
            )
        )
        t_deser_start = time.time()
        # bw_ep, _ = ExecutionPlan.deserialize(serialized_bw_ep)
        bw_ep = pickle.loads(serialized_bw_ep)
        t_deser_end = time.time()
        worker_data.logger.debug(
            "Deserialized bw EP for iteration {} in {} seconds.".format(
                current_batch_idx, t_deser_end - t_deser_start
            )
        )
        t_deser_start = time.time()
        # fw_workload_spec, _ = WorkloadSpec.deserialize(
        #     serialized_fw_workload_spec
        # )
        fw_workload_spec = pickle.loads(serialized_fw_workload_spec)
        t_deser_end = time.time()
        worker_data.logger.debug(
            "Deserialized fw workload spec for iteration {} in {} seconds.".format(
                current_batch_idx, t_deser_end - t_deser_start
            )
        )
        t_deser_start = time.time()
        # bw_workload_spec, _ = WorkloadSpec.deserialize(
        #     serialized_bw_workload_spec
        # )
        bw_workload_spec = pickle.loads(serialized_bw_workload_spec)
        t_deser_end = time.time()
        worker_data.logger.debug(
            "Deserialized bw workload spec for iteration {} in {} seconds.".format(
                current_batch_idx, t_deser_end - t_deser_start
            )
        )
        t_collate_start = time.time()
        local_batch = _collate_samples(
            batch,
            padded_seqlens,
            fw_workload_spec,
            input_key,
            node_rank=node_rank,
            node_local_rank=node_local_rank,
            logger=worker_data.logger,
        )
        t_collate_end = time.time()
        worker_data.logger.debug(
            "Collated data for iteration {} in {} seconds.".format(
                current_batch_idx, t_collate_end - t_collate_start
            )
        )

        worker_data.logger.debug(
            "Generated data for iteration {}.".format(current_batch_idx)
        )
        # increment processed batches
        worker_info.dataset.worker_data.processed_batches += 1
    except Exception as e:
        # explicitly log exception here since it will be swallowed by
        # multiprocessing
        worker_data.logger.error("Exception in worker process: {}".format(e))
        worker_data.logger.error(traceback.format_exc())
        raise e

    # For each iteration we need to return the local batch, forward and backward execution plan
    # and the training spec for getting executor.
    return (
        local_batch,
        fw_ep,
        bw_ep,
        fw_workload_spec,
        bw_workload_spec,
        training_spec,
    )


def get_collate_fn(
    input_key: str = "text",
    training_spec: TrainingSpec = None,
    node_rank: int = 0,
    node_local_rank: int = 0,
):
    return partial(
        _collate_fn,
        input_key=input_key,
        training_spec=training_spec,
        node_rank=node_rank,
        node_local_rank=node_local_rank,
    )


class DataloaderArgs:
    def __init__(
        self,
        dataset,
        batch_size,
        shuffle,
        sampler,
        batch_sampler,
        num_workers,
        drop_last,
        prefetch_factor,
        persistent_workers,
        *args,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.sampler = sampler
        self.batch_sampler = batch_sampler
        self.num_workers = num_workers
        self.drop_last = drop_last
        self.args = args
        self.prefetch_factor = prefetch_factor
        self.persistent_workers = persistent_workers


def _preprocessor_poller(
    control_queue: mp.Queue,
    dataloader_args: DataloaderArgs,
    training_spec: TrainingSpec,
    node_rank,
    node_size,
    is_kv_host,
    assigned_iters_per_node,
    input_key,
    is_dryrun: Optional[bool] = False,
    mask_fn: Optional[Callable] = None,
    mask_fn_extra_keys: Optional[List[str]] = None,
):
    # this process runs only once per physical node, responsible for
    # initializing and polling the preprocessor processes
    torch.set_default_device("cpu")
    logger = create_logger(
        "poller",
        prefix="Poller",
        log_file=f"poller/poller_r{node_rank}.log",
    )
    kv_buffer_size = assigned_iters_per_node * node_size
    logger.debug("Starting poller process.")
    if is_kv_host:
        logger.debug("Starting kvstore server.")
        kv_store, _, _ = _init_kv_store(logger=logger)
        # set up kv_store values for workers
        kv_store.set(
            KVStoreMetaKeys.DCP_SIZE,
            str(training_spec.dcp_size),
            logger=logger,
        )
        kv_store.set(
            KVStoreMetaKeys.TP_SIZE,
            str(training_spec.tensor_parallel_size),
            logger=logger,
        )
        kv_store.set(
            KVStoreMetaKeys.MODEL_SPEC,
            training_spec.model_spec.serialize().decode("iso-8859-1"),
        )
        kv_store.set(
            KVStoreMetaKeys.EXEC_CONTEXT,
            training_spec.exec_context.serialize().decode("iso-8859-1"),
        )
        kv_store.set(KVStoreMetaKeys.MASK_TYPE, training_spec.mask_type)
        kv_store.set(KVStoreMetaKeys.MASK_DTYPE, str(training_spec.mask_dtype))
        kv_store.set(KVStoreMetaKeys.QKV_DTYPE, str(training_spec.qkv_dtype))
        kv_store.set(KVStoreMetaKeys.BLOCK_SIZE, str(training_spec.block_size))
        kv_store.set(
            KVStoreMetaKeys.HEAD_BLOCK_SIZE,
            str(training_spec.head_block_size),
            logger=logger,
        )
        kv_store.set(
            KVStoreMetaKeys.EXEC_CONTEXT,
            training_spec.exec_context.serialize().decode("iso-8859-1"),
            logger=logger,
        )
        kv_store.set(
            KVStoreMetaKeys.KV_BUFFER_SIZE,
            str(kv_buffer_size),
            logger=logger,
        )
        kv_store.set(
            KVStoreMetaKeys.ASSIGNED_ITER_PER_NODE,
            str(assigned_iters_per_node),
            logger=logger,
        )
        kv_store.set(
            KVStoreMetaKeys.USE_BLOCK_SIZE_HEURISTIC,
            str(training_spec.use_block_size_heuristic),
            logger=logger,
        )
        kv_store.set(
            KVStoreMetaKeys.MEM_IMBALANCE_EPSILON,
            str(training_spec.mem_imbalance_epsilon),
            logger=logger,
        )
        kv_store.set(
            KVStoreMetaKeys.COMP_IMBALANCE_EPSILON,
            str(training_spec.comp_imbalance_epsilon),
            logger=logger,
        )
        kv_store.set(
            KVStoreMetaKeys.INTER_NODE_COMP_IMBALANCE_FACTOR,
            str(training_spec.inter_node_comp_imbalance_factor),
            logger=logger,
        )

        # init all ack keys
        for dcp_idx in range(training_spec.dcp_size):
            for i in range(kv_buffer_size):
                kv_store.add(f"r{dcp_idx}_fw_ep_{i}_ack", 1)
                kv_store.add(f"r{dcp_idx}_bw_ep_{i}_ack", 1)
                kv_store.add(f"r{dcp_idx}_fw_wl_{i}_ack", 1)
                kv_store.add(f"r{dcp_idx}_bw_wl_{i}_ack", 1)
                kv_store.add(f"r{dcp_idx}_sl_{i}_ack", 1)

            # set reader ack keys
            for i in range(
                training_spec.pipeline_parallel_size
                * training_spec.tensor_parallel_size
                # * training_spec.n_chunks_per_device
            ):
                for buffer_idx in range(kv_buffer_size):
                    for key in ["ep", "wl"]:
                        kv_store.add(
                            f"r{dcp_idx}_fw_{key}_{buffer_idx}_r{i}_ack", 1
                        )
                        kv_store.add(
                            f"r{dcp_idx}_bw_{key}_{buffer_idx}_r{i}_ack", 1
                        )

                    for key in ["sl"]:
                        kv_store.add(
                            f"r{dcp_idx}_{key}_{buffer_idx}_r{i}_ack", 1
                        )

    # create dataloader
    dataset = dataloader_args.dataset
    # add worker data to it
    preprocess_worker_data = PlannerWorkerData(
        node_rank=node_rank,
        dcp_size=training_spec.dcp_size,
        mask_fn=mask_fn,
        is_dryrun=is_dryrun,
    )
    dataset.worker_data = preprocess_worker_data
    prefetch_data_loader = _DataLoader(
        dataset,
        dataloader_args.batch_size,
        dataloader_args.shuffle,
        dataloader_args.sampler,
        dataloader_args.batch_sampler,
        dataloader_args.num_workers,
        get_preprocessing_collate_fn(
            input_key=input_key,
            mask_fn_extra_keys=mask_fn_extra_keys,
        ),
        False,
        dataloader_args.drop_last,
        0,
        _preprocessing_worker_init_fn,
        *dataloader_args.args,
        prefetch_factor=dataloader_args.prefetch_factor,
        persistent_workers=dataloader_args.persistent_workers,
    )
    # start prefetching
    for idx, _ in enumerate(prefetch_data_loader):
        # try to see if we receive a exit signal
        logger.debug(f"Poller polled for iteration {idx}.")
        try:
            item = control_queue.get_nowait()
            if item == "exit":
                logger.debug("Got exit signal! Poller exiting.")
                break
        except Empty:
            pass
    if is_kv_host:
        kv_store.set(KVStoreMetaKeys.N_ITERS, str(idx + 1), logger=logger)
    logger.debug("No more data to prefetch. Poller exiting.")


def get_num_iters():
    global _kvstore_handle
    if _kvstore_handle is None:
        kv_store, _, _ = _init_kv_store(is_master=False)
        _kvstore_handle = kv_store
    n_iters = _kvstore_handle.get(KVStoreMetaKeys.N_ITERS, wait=False)
    if n_iters is None:
        return None
    return int(n_iters.decode())


def generate_workload(
    q: torch.Tensor,
    kv: torch.Tensor,
    seq_lens: List[int],
    block_size: int,
    head_block_size: int,
    n_total_devices: int,
    n_devices_per_node: int,
    comp_cost_model: AttnRooflineCostModel,
    attn_mask: Optional[torch.Tensor] = None,
    causal: bool = False,
    logger: Optional[logging.Logger] = None,
) -> WorkloadSpec:
    padded_seqlens = [
        seq_len + (block_size - seq_len % block_size) % block_size
        for seq_len in seq_lens
    ]
    return _generate_workload(
        q.shape,
        kv.shape,
        q.dtype,
        seq_lens,
        padded_seqlens,
        block_size,
        head_block_size,
        n_total_devices,
        n_devices_per_node,
        comp_cost_model,
        attn_mask,
        causal,
        logger,
    )


def _generate_workload(
    q_shape,
    kv_shape,
    qkv_dtype,
    raw_seqlens: List[int],
    padded_seqlens: List[int],
    block_size: int,
    head_block_size: int,
    n_total_devices: int,
    n_devices_per_node: int,
    comp_cost_model: AttnRooflineCostModel,
    attn_mask: Optional[torch.Tensor] = None,
    causal: bool = False,
    logger: Optional[logging.Logger] = None,
):
    """
    qkv: torch.Tensor
        qkv tensor with shape (sum(seq_lens), 3, nheads, d)
    attn_mask: torch.Tensor
        attention mask tensor, with shape (sum(seq_lens), 2) or (sum(seq_lens), 2, 2)
        specifying the range of kvs to attend to for each query
    """
    # # first check if input is cached
    # if attn_mask is not None:
    #     assert attn_mask.dtype == torch.int32, "Expected attn_mask to be int32"
    # # first check if input is cached
    # key = (
    #     tuple(qkv_shape),
    #     qkv_dtype,
    #     tuple(seq_lens),
    #     block_size,
    #     head_block_size,
    #     n_total_devices,
    #     n_devices_per_node,
    #     comp_cost_model.max_tflops,
    #     comp_cost_model.max_mem_bw,
    #     _mask_to_tuple(attn_mask),
    #     causal,
    # )
    # key = str(key)
    # with shelve.open(_DCP_WORKLOAD_CACHE_FILE) as cache:
    #     if key in cache:
    #         return cache[key]

    workloads = []
    work_unit_input_map = []
    work_unit_output_map = []

    input_id_to_device = {}
    output_id_to_device = {}
    colocation_constraints = []

    n_heads = q_shape[-2]
    n_query_groups = kv_shape[-2]

    assert n_heads % n_query_groups == 0, (
        f"Number of heads ({n_heads}) must be divisible by number of "
        f"query groups ({n_query_groups})"
    )
    heads_per_query_group = n_heads // n_query_groups
    if logger:
        logger.debug(
            f"Number of heads: {n_heads}, Number of query groups: {n_query_groups}, "
            f"heads per query group: {heads_per_query_group}"
        )
    is_gqa = heads_per_query_group > 1

    # check head block size requirement
    if is_gqa:
        # for gqa, we always use head block size kv = 1,
        # and heads_per_query_group must be divisible by head block size
        assert head_block_size <= heads_per_query_group, (
            f"Head block size ({head_block_size}) must be less than or equal to "
            f"heads per query group ({heads_per_query_group})"
        )
        assert heads_per_query_group % head_block_size == 0, (
            f"# heads per query group ({heads_per_query_group}) must be divisible by "
            f"head block size ({head_block_size})"
        )

    head_block_size_kv = 1 if is_gqa else head_block_size

    assert (
        n_heads % head_block_size == 0
    ), f"Number of heads ({n_heads}) must be divisible by head block size ({head_block_size})"

    n_head_blocks_q = n_heads // head_block_size
    n_head_blocks_kv = n_query_groups // head_block_size_kv

    block_sizes = [block_size] * len(padded_seqlens)

    def _get_block_size(seq_id):
        return block_sizes[seq_id]

    def _get_num_blocks(seq_id):
        return (
            padded_seqlens[seq_id] + block_sizes[seq_id] - 1
        ) // block_sizes[seq_id]

    def _get_actual_seqlen(seq_id, block_id):
        return min(
            block_sizes[seq_id],
            raw_seqlens[seq_id] - block_id * block_sizes[seq_id],
        )

    seq_lengths_in_blocks = [
        _get_num_blocks(seq_id) for seq_id in range(len(padded_seqlens))
    ]

    input_id_to_meta: Dict[int, DataBlockMeta] = {}
    output_id_to_meta: Dict[int, DataBlockMeta] = {}
    work_id_to_meta: Dict[int, BlockMeta] = {}

    buffer_type_to_dtype = {
        BufferType.LOCAL_KV: torch_dtype_to_int(qkv_dtype),
        BufferType.LOCAL_Q: torch_dtype_to_int(qkv_dtype),
        BufferType.LOCAL_OUT: torch_dtype_to_int(qkv_dtype),
        BufferType.LOCAL_LSE: DType.FLOAT32,
        BufferType.LOCAL_dOUT: torch_dtype_to_int(qkv_dtype),
        BufferType.BUFFER_Q: torch_dtype_to_int(qkv_dtype),
        BufferType.BUFFER_KV: torch_dtype_to_int(qkv_dtype),
        BufferType.BUFFER_OUT: torch_dtype_to_int(qkv_dtype),
        BufferType.BUFFER_LSE: DType.FLOAT32,
    }

    cu_padded_seqlens = np.cumsum([0] + padded_seqlens).tolist()
    cumulated_input_units = 0
    cumulated_output_units = 0
    for seq_id, (seq_len_in_blocks, block_size) in enumerate(
        zip(seq_lengths_in_blocks, block_sizes),
    ):
        curr_seq_q_indices = [[] for _ in range(seq_len_in_blocks)]
        curr_seq_kv_indices = [[] for _ in range(seq_len_in_blocks)]
        curr_seq_out_indices = [[] for _ in range(seq_len_in_blocks)]
        curr_seq_lse_indices = [[] for _ in range(seq_len_in_blocks)]
        mask_cache = {}
        for j, i in product(range(n_head_blocks_q), range(seq_len_in_blocks)):
            input_id = cumulated_input_units + j * seq_len_in_blocks + i
            q_meta = DataBlockMeta(
                BlockType.Q,
                torch_dtype_to_int(qkv_dtype),
                seq_id,
                j,
                i,
                _get_actual_seqlen(seq_id, i),
                block_size,
                head_block_size,
                (head_block_size, q_shape[-1]),
            )
            curr_seq_q_indices[i].append(input_id)
            input_id_to_meta[input_id] = q_meta
        for j, i in product(range(n_head_blocks_kv), range(seq_len_in_blocks)):
            input_id = (
                cumulated_input_units
                + n_head_blocks_q * seq_len_in_blocks
                + j * seq_len_in_blocks
                + i
            )
            kv_meta = DataBlockMeta(
                BlockType.KV,
                torch_dtype_to_int(qkv_dtype),
                seq_id,
                j,
                i,
                _get_actual_seqlen(seq_id, i),
                block_size,
                head_block_size_kv,
                (2, head_block_size_kv, kv_shape[-1]),
            )
            curr_seq_kv_indices[i].append(input_id)
            input_id_to_meta[input_id] = kv_meta
        # output sizes: one per work unit, filled during the loop
        for k, i in product(range(n_head_blocks_q), range(seq_len_in_blocks)):
            output_qkv_id = (
                cumulated_output_units + k * seq_len_in_blocks * 2 + (i * 2)
            )
            output_lse_id = (
                cumulated_output_units
                + k * seq_len_in_blocks * 2
                + (i * 2 + 1)
            )
            output_qkv_meta = DataBlockMeta(
                BlockType.Out,
                torch_dtype_to_int(qkv_dtype),
                seq_id,
                k,
                i,
                _get_actual_seqlen(seq_id, i),
                block_size,
                head_block_size,
                (head_block_size, q_shape[-1]),
            )
            output_lse_meta = DataBlockMeta(
                BlockType.LSE,
                torch_dtype_to_int(torch.float32),
                seq_id,
                k,
                i,
                _get_actual_seqlen(seq_id, i),
                block_size,
                head_block_size,
                (head_block_size,),
            )
            output_id_to_meta[output_qkv_id] = output_qkv_meta
            output_id_to_meta[output_lse_id] = output_lse_meta
            curr_seq_out_indices[i].append(output_qkv_id)
            curr_seq_lse_indices[i].append(output_lse_id)
            for j in range(i + 1) if causal else range(seq_len_in_blocks):
                if (i, j) in mask_cache:
                    local_attn_mask = mask_cache[(i, j)]
                elif attn_mask is not None:
                    # test if mask is all empty, otherwise skip
                    local_attn_mask = [-1] * block_size
                    local_attn_mask = [local_attn_mask, local_attn_mask]
                    if attn_mask.dim() == 3:
                        local_attn_mask = [local_attn_mask, local_attn_mask]

                    local_attn_mask = torch.tensor(
                        local_attn_mask, dtype=torch.int32, device="cpu"
                    )
                    # indexing, should be low cost
                    local_seq_mask = attn_mask[
                        cu_padded_seqlens[seq_id]
                        + i
                        * block_size : min(
                            cu_padded_seqlens[seq_id] + (i + 1) * block_size,
                            cu_padded_seqlens[seq_id + 1],
                        ),
                    ]
                    min_col_id = j * block_size
                    max_col_id = min(
                        (j + 1) * block_size, padded_seqlens[seq_id]
                    )
                    if attn_mask.dim() == 2:
                        # a single range
                        min_mask_range = local_seq_mask[:, 0].min().item()
                        max_mask_range = local_seq_mask[:, 1].max().item()
                        if (
                            min_mask_range >= max_col_id
                            or max_mask_range <= min_col_id
                        ):
                            # no overlap, skip
                            continue
                        else:
                            # fill in local_attn_mask, offset by min_col_id
                            # start_time = time.time()
                            start = torch.clamp(
                                torch.clamp(
                                    local_seq_mask[:, 0], min=min_col_id
                                )
                                - min_col_id,
                                min=0,
                            )
                            end = torch.clamp(
                                torch.clamp(
                                    local_seq_mask[:, 1], max=max_col_id
                                )
                                - min_col_id,
                                min=0,
                            )
                            local_attn_mask[0][: len(start)] = start
                            local_attn_mask[1][: len(end)] = end
                            # end_time = time.time()
                            # print(
                            #     f"Operation time: {end_time - start_time} seconds",
                            #     flush=True,
                            # )
                    elif attn_mask.dim() == 3:
                        # two ranges
                        range1_valid = (
                            local_seq_mask[:, 0, 0] < local_seq_mask[:, 0, 1]
                        )  # shape: (block_size,)
                        range2_valid = (
                            local_seq_mask[:, 1, 0] < local_seq_mask[:, 1, 1]
                        )  # shape: (block_size,)
                        range1_valid_any = range1_valid.any()
                        range2_valid_any = range2_valid.any()
                        if not range1_valid_any and not range2_valid_any:
                            # no valid range in mask, skip
                            continue
                        min_mask_range1 = (
                            local_seq_mask[range1_valid, 0, 0].min().item()
                            if range1_valid_any
                            else float("inf")
                        )
                        max_mask_range1 = (
                            local_seq_mask[range1_valid, 0, 1].max().item()
                            if range1_valid_any
                            else float("-inf")
                        )
                        min_mask_range2 = (
                            local_seq_mask[range2_valid, 1, 0].min().item()
                            if range2_valid_any
                            else float("inf")
                        )
                        max_mask_range2 = (
                            local_seq_mask[range2_valid, 1, 1].max().item()
                            if range2_valid_any
                            else float("-inf")
                        )
                        col_overlaps_with_range1 = min(
                            max_col_id, max_mask_range1
                        ) > max(min_col_id, min_mask_range1)
                        col_overlaps_with_range2 = min(
                            max_col_id, max_mask_range2
                        ) > max(min_col_id, min_mask_range2)
                        if (
                            not col_overlaps_with_range1
                            and not col_overlaps_with_range2
                        ):
                            # no overlap, skip
                            continue
                        else:
                            # fill in local_attn_mask, offset by min_col_id
                            # start_time = time.time()
                            start1 = (
                                torch.clamp(
                                    local_seq_mask[:, 0, 0],
                                    min=min_col_id,
                                    max=max_col_id,
                                )
                                - min_col_id
                            )
                            end1 = (
                                torch.clamp(
                                    local_seq_mask[:, 0, 1],
                                    min=min_col_id,
                                    max=max_col_id,
                                )
                                - min_col_id
                            )
                            start2 = (
                                torch.clamp(
                                    local_seq_mask[:, 1, 0],
                                    min=min_col_id,
                                    max=max_col_id,
                                )
                                - min_col_id
                            )
                            end2 = (
                                torch.clamp(
                                    local_seq_mask[:, 1, 1],
                                    min=min_col_id,
                                    max=max_col_id,
                                )
                                - min_col_id
                            )
                            local_attn_mask[0][0][: len(start1)] = start1
                            local_attn_mask[0][1][: len(end1)] = end1
                            local_attn_mask[1][0][: len(start2)] = start2
                            local_attn_mask[1][1][: len(end2)] = end2
                            # end_time = time.time()
                            # print(
                            #     f"Operation time: {end_time - start_time} seconds",
                            #     flush=True,
                            # )
                    else:
                        raise ValueError("attn_mask must be 2D or 3D.")
                    mask_cache[(i, j)] = local_attn_mask
                else:
                    local_attn_mask = None
                work_id = len(workloads)

                input_q_id = cumulated_input_units + k * seq_len_in_blocks + i
                if not is_gqa:
                    kv_head_group_id = k
                else:
                    kv_head_group_id = (
                        k * head_block_size
                    ) // heads_per_query_group
                input_kv_id = (
                    cumulated_input_units
                    + n_head_blocks_q * seq_len_in_blocks
                    + kv_head_group_id * seq_len_in_blocks
                    + j
                )
                work_meta = ComputationBlockMeta(
                    BlockType.Work,
                    torch_dtype_to_int(qkv_dtype),
                    seq_id,
                    k,
                    q_id=input_q_id,
                    kv_id=input_kv_id,
                    out_id=output_qkv_id,
                    lse_id=output_lse_id,
                    local_attn_mask=local_attn_mask,
                    # name=f"S{seq_id}H{k}Q{input_id_to_meta[input_q_id].block_id}KV{input_id_to_meta[input_kv_id].block_id}",
                )
                work_id_to_meta[work_id] = work_meta

                # use cost model to get workload
                flops_cost, mem_cost = comp_cost_model.get_cost(
                    [_get_block_size(seq_id)],
                    [_get_block_size(seq_id)],
                    local_attn_mask,
                    n_heads,
                    n_query_groups,
                    q_shape[-1],
                )
                workloads.append(max(flops_cost, mem_cost))

                work_unit_input_map.append(
                    [
                        input_q_id,
                        input_kv_id,
                    ]
                )
                work_unit_output_map.append(
                    [
                        output_qkv_id,
                        output_lse_id,
                    ]
                )
        cumulated_input_units += seq_len_in_blocks * (
            n_head_blocks_q + n_head_blocks_kv
        )
        cumulated_output_units += seq_len_in_blocks * n_head_blocks_q * 2
        assert (
            len(curr_seq_q_indices)
            == len(curr_seq_kv_indices)
            == len(curr_seq_out_indices)
            == len(curr_seq_lse_indices)
            == seq_len_in_blocks
        )
        for q_indices, kv_indices, out_indices, lse_indices in zip(
            curr_seq_q_indices,
            curr_seq_kv_indices,
            curr_seq_out_indices,
            curr_seq_lse_indices,
        ):
            assert (
                len(q_indices)
                == len(out_indices)
                == len(lse_indices)
                == n_head_blocks_q
            )
            assert len(kv_indices) == n_head_blocks_kv
            colocation_constraints.append(
                [q_indices + kv_indices, out_indices + lse_indices]
            )

    block_mapping = BlockMappings(
        input_id_to_meta,
        output_id_to_meta,
        work_id_to_meta,
        {},  # to be filled by compiler
        {},  # to be filled by compiler
        buffer_type_to_dtype,
    )

    workload_spec = WorkloadSpec(
        workloads,
        work_unit_input_map,
        work_unit_output_map,
        block_mapping,
        input_id_to_device,
        output_id_to_device,
        {},  # work_to_device_map, to be filled by compiler
        {},  # work_to_stage_map, to be filled by compiler
        colocation_constraints,
    )
    # with shelve.open(_DCP_WORKLOAD_CACHE_FILE) as cache:
    #     cache[key] = workload_spec
    return workload_spec


def _block_size_heuristic(
    n_devices: int, raw_seqlens: List[int], mem_epsilon: float
):
    def _get_ref_imbalance(block_size):
        blocks = []
        for seqlen in raw_seqlens:
            if seqlen >= block_size:
                blocks.extend([block_size] * (seqlen // block_size))
            if seqlen % block_size > 0:
                blocks.append(seqlen % block_size)
        # try to evenly distribute blocks across devices
        # use greedy algorithm to get a reference
        per_device_sizes = [0] * n_devices
        for block in sorted(blocks, reverse=True):
            min_idx = per_device_sizes.index(min(per_device_sizes))
            per_device_sizes[min_idx] += block
        ref_imbalance = (
            # max(per_device_sizes) / min(per_device_sizes)
            # if min(per_device_sizes) > 0
            # else float("inf")
            max(per_device_sizes)
            / (sum(per_device_sizes) / len(per_device_sizes))
        )
        if min(per_device_sizes) == 0:
            # invalid distribution
            ref_imbalance = float("inf")
        return ref_imbalance

    block_size = max(raw_seqlens)
    block_size = 2 ** math.ceil(math.log2(block_size))

    while (
        _get_ref_imbalance(block_size) > (1 + mem_epsilon) and block_size > 256
    ):
        block_size //= 2
    return max(256, block_size)


class ExecutionPlanner:
    def __init__(
        self,
        node_rank,
        worker_id,
        exec_context: ExecutionContext,
        model_spec: ModelSpec,
        mask_type: str,
        mask_dtype: int,
        qkv_dtype: int,
        block_size: int,
        head_block_size: int,
        use_block_size_heuristic: Optional[bool] = True,
        mem_epsilon: float = 0.2,
        comp_epsilon: float = 0.2,
        inter_node_comp_imbalance_factor: float = 5.0,
        mask_fn: Optional[Callable] = None,
    ):
        self.exec_context = exec_context
        self.model_spec = model_spec
        self.mask_type = mask_type
        self.mask_dtype = int_to_torch_dtype(mask_dtype)
        self.qkv_dtype = int_to_torch_dtype(qkv_dtype)
        self.block_size = block_size
        self.head_block_size = head_block_size
        self.use_block_size_heuristic = use_block_size_heuristic
        self.mem_epsilon = mem_epsilon
        self.comp_epsilon = comp_epsilon
        self.inter_node_comp_imbalance_factor = (
            inter_node_comp_imbalance_factor
        )
        self.mask_fn = mask_fn

        self.comm_cost_model = exec_context.comm_cost_model
        if self.comm_cost_model is None:
            self.comm_cost_model = CommunicationCostModel()

        self.comp_cost_model = exec_context.comp_cost_model
        if self.comp_cost_model is None:
            self.comp_cost_model = AttnRooflineCostModel()

        assert self.mask_type is not None, "mask_type must be specified"
        if self.mask_type != "causal":
            assert (
                self.mask_fn is not None
            ), "mask_fn must be specified if mask_type is not causal"

        self.logger = create_logger(
            name="ExecutionPlanner",
            prefix=f"Node {node_rank} | " f"Planner {worker_id}",
            log_file="planner/" f"nr{node_rank}_p{worker_id}.log",
        )

    def plan(
        self,
        raw_seqlens,
        dropout_p=0.0,
        window_size=(-1, -1),
        reduction_schedule_algo="delayed",
        executor_impl="python",
        deterministic=False,
        iteration_idx=0,
        mask_fn_info=None,
    ) -> Tuple[
        List[ExecutionPlan],
        List[ExecutionPlan],
        WorkloadSpec,
        WorkloadSpec,
        List[int],
    ]:
        # Follow the prepare_dcp_distributed_attn_for_test
        # to generate the executor, execution plan and workload

        self.logger.debug(
            "Received raw seqlens: {}".format(raw_seqlens),
        )

        if self.use_block_size_heuristic:
            self.heuristic_set_block_sizes(raw_seqlens)

        padded_seqlens = self.pad_seqlens(raw_seqlens)

        attn_mask = self.get_mask(raw_seqlens, padded_seqlens, mask_fn_info)
        q_shape, kv_shape = self.get_qkv_shape(raw_seqlens)

        n_total_devices = (
            self.exec_context.n_nodes * self.exec_context.n_devices_per_node
        )
        n_devices_per_node = self.exec_context.n_devices_per_node

        workload_spec = None
        # generate execution plans on master node
        start_time = time.time()
        workload_spec = _generate_workload(
            q_shape,
            kv_shape,
            self.qkv_dtype,
            raw_seqlens,
            padded_seqlens,
            self.block_size,
            self.head_block_size,
            n_total_devices,
            n_devices_per_node,
            self.comp_cost_model,
            attn_mask=attn_mask,
            logger=self.logger,
        )
        end_time = time.time()
        self.logger.debug(
            f"Workload generation took {end_time - start_time} seconds",
        )
        config = CompilerConfig(
            mem_imbalance_epsilon=self.mem_epsilon,
            comp_imbalance_epsilon=self.comp_epsilon,
            inter_node_comp_imbalance_factor=self.inter_node_comp_imbalance_factor,
            reduction_schedule_algo=reduction_schedule_algo,
        )
        compiler = InstrCompiler(
            self.exec_context, config, iteration_idx=iteration_idx
        )
        compiler.logger.debug(
            f"Using block size: {self.block_size}, "
            f"head_block_size: {self.head_block_size}, "
            f"mem_epsilon: {self.mem_epsilon}, "
            f"comp_epsilon: {self.comp_epsilon}, "
            f"inter_node_comp_imbalance_factor: {self.inter_node_comp_imbalance_factor}, "
            f"received padded seqlens: {padded_seqlens}, "
            f"raw_seqlens: {raw_seqlens}"
        )
        start_time = time.time()
        (
            fw_workload,
            bw_workload,
            fw_execution_plans,
            bw_execution_plans,
        ) = compiler.compile(workload_spec, generate_backward=True)
        end_time = time.time()
        self.logger.debug(
            f"Compilation took {end_time - start_time} seconds",
        )
        return (
            fw_execution_plans,
            bw_execution_plans,
            fw_workload,
            bw_workload,
            padded_seqlens,
        )

    def get_qkv_shape(self, raw_seqlens):
        return [
            sum(raw_seqlens),
            self.model_spec.n_heads,
            self.model_spec.head_dim,
        ], [
            sum(raw_seqlens),
            2,
            self.model_spec.n_query_groups,
            self.model_spec.head_dim,
        ]

    def get_mask(self, raw_seqlens, padded_seqlens, mask_fn_info):
        if self.mask_type == "causal":
            attn_mask = torch.zeros(
                sum(padded_seqlens), 2, dtype=self.mask_dtype
            )
            cu_seqlens_padded = F.pad(
                torch.cumsum(
                    torch.tensor(padded_seqlens, dtype=self.mask_dtype), 0
                ),
                (1, 0),
            ).to(self.mask_dtype)
            for seq_id, seqlen in enumerate(raw_seqlens):
                attn_mask[
                    cu_seqlens_padded[seq_id] : cu_seqlens_padded[seq_id]
                    + raw_seqlens[seq_id],
                    0,
                ] = 0
                attn_mask[
                    cu_seqlens_padded[seq_id] : cu_seqlens_padded[seq_id]
                    + raw_seqlens[seq_id],
                    1,
                ] = torch.arange(1, seqlen + 1, dtype=self.mask_dtype)
        elif self.mask_fn is not None:
            attn_mask = self.mask_fn(
                raw_seqlens, padded_seqlens, self.mask_dtype, mask_fn_info
            )
        else:
            raise ValueError(f"Unsupported mask type: {self.mask_type}")

        return attn_mask

    def heuristic_set_block_sizes(self, raw_seqlens):
        n_devices = (
            self.exec_context.n_nodes * self.exec_context.n_devices_per_node
        )

        self.block_size = _block_size_heuristic(
            n_devices, raw_seqlens, self.mem_epsilon
        )

        self.logger.debug(
            f"Setting block size to {self.block_size} based on heuristic."
        )

    def pad_seqlens(self, raw_seqlens):
        # Make all seqlen multiple times of block size
        return [
            (seqlen + self.block_size - 1) // self.block_size * self.block_size
            for seqlen in raw_seqlens
        ]


class DCPDataLoader:
    """
    A wrapper around PyTorch DataLoader, which automatically generates
    execution plans for each batch and returns the execution plan along
    with the batch of data.

    On local rank 0 of each node, it starts a poller process which creates
    a Torch DataLoader wrapping the user provided dataset and prefetches data.
    Each worker in the Torch DataLoader is instructed to compute the execution
    plan for assigned batches and pushes the execution plan to a shared kv
    store. On the node where kv store is hosted, it is also responsible for kv
    store initialization.

    On all ranks, it creates a torch DataLoader wrapping the user dataset.
    In addition to the data, it also returns the execution plan for the batch,
    fetched from the shared kv store.
    """

    def __init__(
        self,
        training_spec: TrainingSpec,
        dataset,
        is_kv_host,
        node_rank=0,
        node_local_rank=0,
        node_size=1,
        dcp_rank=0,
        pp_rank=0,
        tp_rank=0,
        batch_size=1,
        start_poller=False,
        shuffle=False,
        sampler=None,
        batch_sampler=None,
        num_workers=0,
        num_preprocess_workers=32,
        pin_memory=False,
        drop_last=False,
        timeout=0,
        *args,
        prefetch_factor=2,
        persistent_workers=False,
        input_key="text",
        mask_fn: Optional[Callable] = None,
        mask_fn_extra_keys: Optional[List[str]] = None,
        is_dryrun=False,
    ):
        self.node_rank = node_rank
        self.node_local_rank = node_local_rank
        assert isinstance(node_rank, int), "node_rank must be an integer."
        assert isinstance(
            node_local_rank, int
        ), "node_local_rank must be an integer."
        self.dcp_rank = dcp_rank
        self.pp_rank = pp_rank
        self.tp_rank = tp_rank
        assert pp_rank < training_spec.pipeline_parallel_size, (
            f"pp_rank ({pp_rank}) should be smaller than "
            f"pipeline_parallel_size ({training_spec.pipeline_parallel_size})"
            "in training_spec."
        )
        # create queues
        self.poller_control_queue = mp.Queue()
        self.num_preprocess_workers = num_preprocess_workers

        if is_kv_host:
            # start kv store server
            self.kv_server, _, _ = _init_kv_server(logger=logger)

        if start_poller:
            dataloader_args = DataloaderArgs(
                dataset,
                batch_size,
                shuffle,
                sampler,
                batch_sampler,
                num_preprocess_workers,
                drop_last,
                prefetch_factor,
                persistent_workers,
                *args,
            )
            assigned_iters_per_node = num_preprocess_workers * prefetch_factor
            self.poller_process = mp.Process(
                target=_preprocessor_poller,
                args=(
                    self.poller_control_queue,
                    dataloader_args,
                    training_spec,
                    node_rank,
                    node_size,
                    is_kv_host,
                    assigned_iters_per_node,
                    input_key,
                    is_dryrun,
                    mask_fn,
                    mask_fn_extra_keys,
                ),
            )
            self.poller_process.start()
        # create torch dataloader
        worker_data = DataloaderWorkerData(
            dcp_rank=dcp_rank,
            pp_rank=pp_rank,
            tp_rank=tp_rank,
            is_dryrun=is_dryrun,
        )
        dataset.worker_data = worker_data
        self.data_loader = _DataLoader(
            dataset,
            batch_size,
            shuffle,
            sampler,
            batch_sampler,
            num_workers,
            get_collate_fn(
                input_key,
                training_spec=training_spec,
                node_rank=self.node_rank,
                node_local_rank=self.node_local_rank,
            ),
            pin_memory,
            drop_last,
            timeout,
            _worker_init_fn,
            *args,
            prefetch_factor=prefetch_factor,
            persistent_workers=persistent_workers,
        )

    def __del__(self):
        if hasattr(self, "poller_process"):
            if self.poller_process.is_alive():
                self.poller_control_queue.put("exit")
                self.poller_process.join()

    def __iter__(self):
        yield from self.data_loader

    def __len__(self):
        return self.data_loader.__len__()

    def __next__(self):
        if not hasattr(self, "data_loader_iter"):
            self.data_loader_iter = iter(self.data_loader)

        return next(self.data_loader_iter)

    def check_worker_number_rationality(self):
        if self.num_preprocess_workers == 0:
            logger.warning(
                "DCPDataLoader should be used with a large number of "
                "preprocessing workers to achieve good performance. "
                "Current setting is num_preprocess_workers=0."
            )
        self.data_loader.check_worker_number_rationality()
