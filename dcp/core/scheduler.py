import os
import time
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
from itertools import product
from typing import Any, Dict, List, Optional, Tuple

import torch

from dcp.core.block_table import (
    BlockManager,
    BlockMappings,
    BlockType,
    ComputationBlockMeta,
    DataBlockMapping,
    WorkloadSpec,
)
from dcp.core.common import ExecutionContext
from dcp.core.cost_model import CommunicationCostModel
from dcp.core.instructions import (
    BarrierInstr,
    AttnBackwardInstr,
    AttnInstr,
    AttnReductionInstr,
    BufferBlock,
    BufferInfo,
    BufferType,
    CommLaunchInstr,
    CommOp,
    CommType,
    CommWaitInstr,
    ExecutionPlan,
    MemcpyInstr,
    ReductionOp,
    SumInstr,
    SumOp,
)
from dcp.core.pipeline import (
    check_and_visualize_pipeline,
    generate_pipelines_n_stages,
    generate_pipelines_n_stages_with_tree_packing,
    log_pipeline_comm_cost,
)
from dcp.utils.common import read_env_int
from dcp.utils.logger import Logger, read_env_bool

SCHEDULE_LOGGING_ENABLED = read_env_bool("DCP_LOG_SCHEDULE", default=False)
SCHEDULE_DUMP_PIPELINE = read_env_bool("DCP_DUMP_PIPELINE", default=False)
SCHEDULE_LOG_BUFFER_INFO = read_env_bool("DCP_LOG_BUFFER_INFO", default=False)
SCHEDULE_DISABLE_TWO_PHASE_COMM = read_env_bool(
    "DCP_DISABLE_TWO_PHASE_COMM", default=False
)

SCHEDULE_FORCE_SINGLE_STAGE = read_env_bool(
    "DCP_FORCE_SINGLE_STAGE", default=False
)
SCHEDULE_N_STAGES = read_env_int("DCP_N_STAGES", default=4)

_UNPADDED_BUFFER_TYPES = {
    BufferType.LOCAL_Q,
    BufferType.LOCAL_dQ,
    BufferType.LOCAL_KV,
    BufferType.LOCAL_dKV,
    BufferType.LOCAL_OUT,
    BufferType.LOCAL_dOUT,
    BufferType.LOCAL_LSE,
}
_QKV_ORDERED_BUFFER_BLOCK_TYPES = {
    BlockType.Q,
    BlockType.dQ,
    BlockType.KV,
    BlockType.dKV,
    BlockType.Out,
    BlockType.dOut,
}
_LSE_ORDERED_BUFFER_BLOCK_TYPES = {
    BlockType.LSE,
}
_LOCAL_BLOCK_TYPE_TO_BUFFER_TYPE = {
    BlockType.Q: BufferType.LOCAL_Q,
    BlockType.dQ: BufferType.LOCAL_dQ,
    BlockType.KV: BufferType.LOCAL_KV,
    BlockType.dKV: BufferType.LOCAL_dKV,
    BlockType.Out: BufferType.LOCAL_OUT,
    BlockType.dOut: BufferType.LOCAL_dOUT,
    BlockType.LSE: BufferType.LOCAL_LSE,
}
_BUFFER_BLOCK_TYPE_TO_BUFFER_TYPE = {
    BlockType.Q: BufferType.BUFFER_Q,
    BlockType.dQ: BufferType.BUFFER_dQ,
    BlockType.KV: BufferType.BUFFER_KV,
    BlockType.dKV: BufferType.BUFFER_dKV,
    BlockType.Out: BufferType.BUFFER_OUT,
    BlockType.dOut: BufferType.BUFFER_dOUT,
    BlockType.LSE: BufferType.BUFFER_LSE,
}
_BUFFER_TYPE_TO_BLOCK_TYPE = {
    BufferType.BUFFER_Q: BlockType.Q,
    BufferType.BUFFER_dQ: BlockType.dQ,
    BufferType.BUFFER_KV: BlockType.KV,
    BufferType.BUFFER_dKV: BlockType.dKV,
    BufferType.BUFFER_OUT: BlockType.Out,
    BufferType.BUFFER_dOUT: BlockType.dOut,
    BufferType.BUFFER_LSE: BlockType.LSE,
}
_FW_BLOCK_INPUT_TYPES = {
    BlockType.Q,
    BlockType.KV,
}
_FW_BLOCK_OUTPUT_TYPES = {
    BlockType.Out,
    BlockType.LSE,
}
_BW_BLOCK_INPUT_TYPES = {
    BlockType.Q,
    BlockType.KV,
    BlockType.Out,
    BlockType.dOut,
    BlockType.LSE,
}
_BW_BLOCK_OUTPUT_TYPES = {
    BlockType.dQ,
    BlockType.dKV,
}


@dataclass
class RevBlockMapping:
    n_total_sequences: int
    n_head_blocks: int
    n_kv_head_blocks: int
    n_blocks_per_sequence: List[int]
    seq_head_block_id_to_q_input_id: Dict[Tuple[int, int, int], int]
    seq_head_block_id_to_kv_input_id: Dict[Tuple[int, int, int], int]
    seq_head_block_id_to_output_id: Dict[Tuple[int, int, int], int]
    seq_head_block_id_to_lse_id: Dict[Tuple[int, int, int], int]
    seq_id_to_work_ids: Dict[int, List[Tuple[int, int, int, int, int, int]]]


def link_sequence_block_to_input_output_id(
    workload_spec: WorkloadSpec,
) -> RevBlockMapping:
    n_total_sequences = (
        max(
            v.seq_id
            for v in workload_spec.block_mapping.input_id_to_meta.values()
        )
        + 1
    )
    n_head_blocks = (
        max(
            v.head_id
            for v in workload_spec.block_mapping.input_id_to_meta.values()
            if v.type == BlockType.Q
        )
        + 1
    )
    n_kv_head_blocks = (
        max(
            v.head_id
            for v in workload_spec.block_mapping.input_id_to_meta.values()
            if v.type == BlockType.KV
        )
        + 1
    )
    n_blocks_per_sequence = []
    for seq_id in range(n_total_sequences):
        n_blocks_per_sequence.append(
            max(
                v.block_id
                for v in workload_spec.block_mapping.input_id_to_meta.values()
                if v.seq_id == seq_id
            )
            + 1
        )

    seq_id_to_work_ids = defaultdict(list)
    for work_id in range(len(workload_spec.workloads)):
        meta: ComputationBlockMeta = (
            workload_spec.block_mapping.work_id_to_meta[work_id]
        )
        seq_id = meta.seq_id
        head_id = meta.head_id
        q_id = meta.q_id
        kv_id = meta.kv_id
        out_id = meta.out_id
        lse_id = meta.lse_id
        seq_id_to_work_ids[seq_id].append(
            (work_id, head_id, q_id, kv_id, out_id, lse_id)
        )

    seq_head_block_id_to_q_input_id = {}
    seq_head_block_id_to_kv_input_id = {}
    for (
        input_id,
        meta,
    ) in workload_spec.block_mapping.input_id_to_meta.items():
        if meta.type == BlockType.Q:
            seq_head_block_id_to_q_input_id[
                (meta.seq_id, meta.head_id, meta.block_id)
            ] = input_id
        elif meta.type == BlockType.KV:
            seq_head_block_id_to_kv_input_id[
                (meta.seq_id, meta.head_id, meta.block_id)
            ] = input_id
    seq_head_block_id_to_output_id = {}
    seq_head_block_id_to_lse_id = {}
    for (
        output_id,
        meta,
    ) in workload_spec.block_mapping.output_id_to_meta.items():
        if meta.type == BlockType.Out:
            seq_head_block_id_to_output_id[
                (meta.seq_id, meta.head_id, meta.block_id)
            ] = output_id
        elif meta.type == BlockType.LSE:
            seq_head_block_id_to_lse_id[
                (meta.seq_id, meta.head_id, meta.block_id)
            ] = output_id
    return RevBlockMapping(
        n_total_sequences=n_total_sequences,
        n_head_blocks=n_head_blocks,
        n_kv_head_blocks=n_kv_head_blocks,
        n_blocks_per_sequence=n_blocks_per_sequence,
        seq_head_block_id_to_q_input_id=seq_head_block_id_to_q_input_id,
        seq_head_block_id_to_kv_input_id=seq_head_block_id_to_kv_input_id,
        seq_head_block_id_to_output_id=seq_head_block_id_to_output_id,
        seq_head_block_id_to_lse_id=seq_head_block_id_to_lse_id,
        seq_id_to_work_ids=seq_id_to_work_ids,
    )


def convert_forward_to_backward(
    fw_workload_spec: WorkloadSpec,
    bw_workloads: List[int],
    logger: Optional[Logger] = None,
):
    """
    work_id is still the same
    FW: q, kv -> out, lse
    BW: q, kv, out, dout, lse -> dq, dkv
    """
    # step 1, reindex inputs and outputs
    fw_workload_spec = deepcopy(fw_workload_spec)
    rev_block_mapping = link_sequence_block_to_input_output_id(
        fw_workload_spec
    )

    bw_input_id_to_meta: DataBlockMapping = {}
    bw_output_id_to_meta: DataBlockMapping = {}
    bw_work_id_to_meta: Dict[int, ComputationBlockMeta] = {}
    bw_device_to_input_map = defaultdict(list)
    bw_device_to_output_map = defaultdict(list)

    fw_q_id_to_bw_q_id = {}
    fw_kv_id_to_bw_kv_id = {}
    fw_out_id_to_bw_out_id = {}
    fw_lse_id_to_bw_lse_id = {}
    bw_seq_head_block_id_to_dout_id = {}
    bw_seq_head_block_id_to_dq_id = {}
    fw_kv_id_to_bw_dkv_id = {}

    assert (
        rev_block_mapping.n_head_blocks % rev_block_mapping.n_kv_head_blocks
        == 0
    ), (
        f"n_kv_head_blocks {rev_block_mapping.n_kv_head_blocks} "
        f"must be divisible by n_head_blocks {rev_block_mapping.n_head_blocks}"
    )
    h_h_kv_ratio = (
        rev_block_mapping.n_head_blocks // rev_block_mapping.n_kv_head_blocks
    )

    for seq_id, head_id in product(
        range(rev_block_mapping.n_total_sequences),
        range(rev_block_mapping.n_head_blocks),
    ):
        for block_id in range(rev_block_mapping.n_blocks_per_sequence[seq_id]):
            kv_head_id = head_id // h_h_kv_ratio
            fw_q_id = rev_block_mapping.seq_head_block_id_to_q_input_id[
                (seq_id, head_id, block_id)
            ]
            fw_q_meta = fw_workload_spec.block_mapping.input_id_to_meta[
                fw_q_id
            ]
            fw_q_device = fw_workload_spec.input_to_device_map[fw_q_id]
            fw_kv_id = rev_block_mapping.seq_head_block_id_to_kv_input_id[
                (seq_id, kv_head_id, block_id)
            ]
            fw_kv_meta = fw_workload_spec.block_mapping.input_id_to_meta[
                fw_kv_id
            ]
            fw_kv_device = fw_workload_spec.input_to_device_map[fw_kv_id]
            fw_out_id = rev_block_mapping.seq_head_block_id_to_output_id[
                (seq_id, head_id, block_id)
            ]
            fw_out_meta = fw_workload_spec.block_mapping.output_id_to_meta[
                fw_out_id
            ]
            fw_out_device = fw_workload_spec.output_to_device_map[fw_out_id]
            fw_lse_id = rev_block_mapping.seq_head_block_id_to_lse_id[
                (seq_id, head_id, block_id)
            ]
            fw_lse_meta = fw_workload_spec.block_mapping.output_id_to_meta[
                fw_lse_id
            ]
            fw_lse_device = fw_workload_spec.output_to_device_map[fw_lse_id]

            bw_q_id = len(bw_input_id_to_meta)
            bw_q_meta = fw_q_meta
            bw_input_id_to_meta[bw_q_id] = bw_q_meta
            bw_device_to_input_map[fw_q_device].append(bw_q_id)
            fw_q_id_to_bw_q_id[fw_q_id] = bw_q_id

            if fw_kv_id not in fw_kv_id_to_bw_kv_id:
                bw_kv_id = len(bw_input_id_to_meta)
                bw_kv_meta = fw_kv_meta
                bw_input_id_to_meta[bw_kv_id] = bw_kv_meta
                bw_device_to_input_map[fw_kv_device].append(bw_kv_id)
                fw_kv_id_to_bw_kv_id[fw_kv_id] = bw_kv_id

            bw_out_id = len(bw_input_id_to_meta)
            bw_out_meta = fw_out_meta
            bw_input_id_to_meta[bw_out_id] = bw_out_meta
            bw_device_to_input_map[fw_out_device].append(bw_out_id)
            fw_out_id_to_bw_out_id[fw_out_id] = bw_out_id

            bw_lse_id = len(bw_input_id_to_meta)
            bw_lse_meta = fw_lse_meta
            bw_input_id_to_meta[bw_lse_id] = bw_lse_meta
            bw_device_to_input_map[fw_lse_device].append(bw_lse_id)
            fw_lse_id_to_bw_lse_id[fw_lse_id] = bw_lse_id

            # dOut
            bw_dout_id = len(bw_input_id_to_meta)
            bw_dout_meta = deepcopy(fw_out_meta)
            bw_dout_meta.type = BlockType.dOut
            bw_input_id_to_meta[bw_dout_id] = bw_dout_meta
            bw_device_to_input_map[fw_out_device].append(bw_dout_id)
            bw_seq_head_block_id_to_dout_id[(seq_id, head_id, block_id)] = (
                bw_dout_id
            )
            # dQ
            bw_dq_id = len(bw_output_id_to_meta)
            bw_dq_meta = deepcopy(fw_q_meta)
            bw_dq_meta.type = BlockType.dQ
            bw_output_id_to_meta[bw_dq_id] = bw_dq_meta
            bw_device_to_output_map[fw_q_device].append(bw_dq_id)
            bw_seq_head_block_id_to_dq_id[(seq_id, head_id, block_id)] = (
                bw_dq_id
            )
            # dKV
            if fw_kv_id not in fw_kv_id_to_bw_dkv_id:
                bw_dkv_id = len(bw_output_id_to_meta)
                bw_dkv_meta = deepcopy(fw_kv_meta)
                bw_dkv_meta.type = BlockType.dKV
                bw_output_id_to_meta[bw_dkv_id] = bw_dkv_meta
                bw_device_to_output_map[fw_kv_device].append(bw_dkv_id)
                fw_kv_id_to_bw_dkv_id[fw_kv_id] = bw_dkv_id

    bw_input_to_device_map = {}
    for d, input_ids in bw_device_to_input_map.items():
        for input_id in input_ids:
            bw_input_to_device_map[input_id] = d
    bw_output_to_device_map = {}
    for d, output_ids in bw_device_to_output_map.items():
        for output_id in output_ids:
            bw_output_to_device_map[output_id] = d

    # fill in workload input and outputs
    bw_work_unit_input_map = [
        [] for _ in range(len(fw_workload_spec.workloads))
    ]
    bw_work_unit_output_map = [
        [] for _ in range(len(fw_workload_spec.workloads))
    ]
    bw_input_id_to_buffer_index = defaultdict(dict)
    bw_output_id_to_buffer_index = defaultdict(dict)
    for workload_id in range(len(fw_workload_spec.workloads)):
        work_meta: ComputationBlockMeta = (
            fw_workload_spec.block_mapping.work_id_to_meta[workload_id]
        )
        seq_id = work_meta.seq_id
        q_id = work_meta.q_id
        kv_id = work_meta.kv_id
        out_id = work_meta.out_id
        lse_id = work_meta.lse_id
        head_id = fw_workload_spec.block_mapping.input_id_to_meta[q_id].head_id
        block_id = fw_workload_spec.block_mapping.input_id_to_meta[
            q_id
        ].block_id

        bw_q_id = fw_q_id_to_bw_q_id[q_id]
        bw_kv_id = fw_kv_id_to_bw_kv_id[kv_id]
        bw_out_id = fw_out_id_to_bw_out_id[out_id]
        bw_lse_id = fw_lse_id_to_bw_lse_id[lse_id]
        bw_dout_id = bw_seq_head_block_id_to_dout_id[
            (seq_id, head_id, block_id)
        ]
        bw_dq_id = bw_seq_head_block_id_to_dq_id[(seq_id, head_id, block_id)]
        bw_dkv_id = fw_kv_id_to_bw_dkv_id[kv_id]

        bw_work_meta = ComputationBlockMeta(
            BlockType.Work,
            work_meta.dtype,
            seq_id,
            head_id,
            bw_q_id,
            bw_kv_id,
            bw_out_id,
            bw_lse_id,
            bw_dout_id,
            bw_dq_id,
            bw_dkv_id,
            work_meta.local_attn_mask,
        )

        bw_work_id_to_meta[workload_id] = bw_work_meta

        bw_work_unit_input_map[workload_id] = [
            bw_q_id,
            bw_kv_id,
            bw_out_id,
            bw_lse_id,
            bw_dout_id,
        ]
        bw_work_unit_output_map[workload_id] = [bw_dq_id, bw_dkv_id]

        fw_q_device = fw_workload_spec.input_to_device_map[q_id]
        fw_q_buffer_index = (
            fw_workload_spec.block_mapping.input_id_to_buffer_index[
                fw_q_device
            ][q_id]
        )
        fw_kv_device = fw_workload_spec.input_to_device_map[kv_id]
        fw_kv_buffer_index = (
            fw_workload_spec.block_mapping.input_id_to_buffer_index[
                fw_kv_device
            ][kv_id]
        )
        fw_out_device = fw_workload_spec.output_to_device_map[out_id]
        fw_out_buffer_index = (
            fw_workload_spec.block_mapping.output_id_to_buffer_index[
                fw_out_device
            ][out_id]
        )
        fw_lse_device = fw_workload_spec.output_to_device_map[lse_id]
        fw_lse_buffer_index = (
            fw_workload_spec.block_mapping.output_id_to_buffer_index[
                fw_lse_device
            ][lse_id]
        )

        bw_input_id_to_buffer_index[fw_q_device][bw_q_id] = fw_q_buffer_index
        bw_input_id_to_buffer_index[fw_kv_device][
            bw_kv_id
        ] = fw_kv_buffer_index
        bw_input_id_to_buffer_index[fw_out_device][
            bw_out_id
        ] = fw_out_buffer_index
        bw_input_id_to_buffer_index[fw_lse_device][
            bw_lse_id
        ] = fw_lse_buffer_index
        bw_input_id_to_buffer_index[fw_out_device][
            bw_dout_id
        ] = fw_out_buffer_index

        bw_output_id_to_buffer_index[fw_q_device][bw_dq_id] = fw_q_buffer_index
        bw_output_id_to_buffer_index[fw_kv_device][
            bw_dkv_id
        ] = fw_kv_buffer_index

    # fill in the buffer type to dtype
    bw_buffer_type_to_dtype = {
        BufferType.BUFFER_Q: fw_workload_spec.block_mapping.buffer_type_to_dtype[
            BufferType.BUFFER_Q
        ],
        BufferType.BUFFER_dQ: fw_workload_spec.block_mapping.buffer_type_to_dtype[
            BufferType.BUFFER_Q
        ],
        BufferType.BUFFER_KV: fw_workload_spec.block_mapping.buffer_type_to_dtype[
            BufferType.BUFFER_KV
        ],
        BufferType.BUFFER_dKV: fw_workload_spec.block_mapping.buffer_type_to_dtype[
            BufferType.BUFFER_KV
        ],
        BufferType.BUFFER_OUT: fw_workload_spec.block_mapping.buffer_type_to_dtype[
            BufferType.BUFFER_OUT
        ],
        BufferType.BUFFER_dOUT: fw_workload_spec.block_mapping.buffer_type_to_dtype[
            BufferType.BUFFER_OUT
        ],
        BufferType.BUFFER_LSE: fw_workload_spec.block_mapping.buffer_type_to_dtype[
            BufferType.BUFFER_LSE
        ],
        BufferType.LOCAL_Q: fw_workload_spec.block_mapping.buffer_type_to_dtype[
            BufferType.LOCAL_Q
        ],
        BufferType.LOCAL_dQ: fw_workload_spec.block_mapping.buffer_type_to_dtype[
            BufferType.LOCAL_Q
        ],
        BufferType.LOCAL_KV: fw_workload_spec.block_mapping.buffer_type_to_dtype[
            BufferType.LOCAL_KV
        ],
        BufferType.LOCAL_dKV: fw_workload_spec.block_mapping.buffer_type_to_dtype[
            BufferType.LOCAL_KV
        ],
        BufferType.LOCAL_OUT: fw_workload_spec.block_mapping.buffer_type_to_dtype[
            BufferType.BUFFER_Q
        ],
        BufferType.LOCAL_dOUT: fw_workload_spec.block_mapping.buffer_type_to_dtype[
            BufferType.BUFFER_Q
        ],
        BufferType.LOCAL_LSE: fw_workload_spec.block_mapping.buffer_type_to_dtype[
            BufferType.BUFFER_LSE
        ],
    }
    bw_block_mapping = BlockMappings(
        bw_input_id_to_meta,
        bw_output_id_to_meta,
        bw_work_id_to_meta,
        bw_input_id_to_buffer_index,
        bw_output_id_to_buffer_index,
        bw_buffer_type_to_dtype,
    )
    bw_workload_spec = WorkloadSpec(
        bw_workloads,
        bw_work_unit_input_map,
        bw_work_unit_output_map,
        bw_block_mapping,
        bw_input_to_device_map,
        bw_output_to_device_map,
        fw_workload_spec.work_to_device_map,
        {},
        [],
    )
    return bw_workload_spec


def schedule(
    context: ExecutionContext,
    workload: WorkloadSpec,
    comm_cost_model: CommunicationCostModel,
    logger: Optional[Logger] = None,
    is_forward: bool = True,
    pipeline_algo: str = "greedy",
):
    logging_enabled = logger is not None and SCHEDULE_LOGGING_ENABLED

    def _logger_debug(msg: str):
        if logging_enabled:
            logger.debug(msg)

    _logger_debug(f"Scheduling {'FORWARD' if is_forward else 'BACKWARD'}.")

    # move some references to local variables
    workload_to_device_map = workload.work_to_device_map
    input_to_device_map = workload.input_to_device_map
    output_to_device_map = workload.output_to_device_map
    # # extract input_sizes from block_mapping
    # input_sizes = []
    # for input_id in sorted(workload.block_mapping.input_id_to_meta.keys()):
    #     input_sizes.append(
    #         workload.block_mapping.input_id_to_meta[input_id].numel(
    #             padded=False
    #         )
    #     )
    all_devices = [
        (node_id, device_id)
        for node_id in range(context.n_nodes)
        for device_id in range(context.n_devices_per_node)
    ]
    # construct pipelines
    n_stages = SCHEDULE_N_STAGES if not SCHEDULE_FORCE_SINGLE_STAGE else 1
    start_time = time.time()
    _logger_debug("Input sizes: {}".format(workload.input_sizes))
    if pipeline_algo == "tree":
        pipeline_fn = generate_pipelines_n_stages_with_tree_packing
    elif pipeline_algo == "greedy":
        pipeline_fn = generate_pipelines_n_stages
    else:
        raise ValueError(f"Unknown pipeline algorithm: {pipeline_algo}")
    workload_per_stage, comms_per_stage, second_phase_comm_per_stage = (
        pipeline_fn(
            n_stages,
            all_devices,
            workload.workloads,
            workload.work_unit_input_map,
            workload.block_mapping.input_id_to_meta,
            workload.input_sizes,
            workload_to_device_map,
            input_to_device_map,
            not is_forward,
            logger=logger,
        )
    )
    if logging_enabled:
        log_pipeline_comm_cost(
            comms_per_stage, workload.input_sizes, logger=logger
        )
    end_time = time.time()
    if logging_enabled:
        _logger_debug(
            f"Pipeline generation took {end_time - start_time} seconds."
        )
        if SCHEDULE_DUMP_PIPELINE:
            _logger_debug("Dumping pipeline to file...")
            check_and_visualize_pipeline(
                workload.workloads,
                workload.input_sizes,
                workload.work_unit_input_map,
                workload_per_stage,
                input_to_device_map,
                comm_cost_model.inter_node_bandwidth,
                comm_cost_model.intra_node_bandwidth,
                out_file=os.path.dirname(logger.log_file) + "/pipeline.json",
            )
        _logger_debug("Workload per stage per device:")
        for stage_id, stage in enumerate(workload_per_stage):
            _logger_debug(f"Stage {stage_id}")
            for d, workload_ids in stage.items():
                _logger_debug(f"\tDevice {d}: {workload_ids}")
    # init block managers, one for each tensor type

    q_block_managers = {d: BlockManager() for d in all_devices}
    kv_block_managers = {d: BlockManager() for d in all_devices}
    out_block_managers = {d: BlockManager() for d in all_devices}
    lse_block_managers = {d: BlockManager() for d in all_devices}

    dout_block_managers = {d: BlockManager() for d in all_devices}
    dq_block_managers = {d: BlockManager() for d in all_devices}
    dkv_block_managers = {d: BlockManager() for d in all_devices}

    _BLOCK_TYPE_TO_MANAGERS = {
        BlockType.Q: q_block_managers,
        BlockType.dQ: dq_block_managers,
        BlockType.KV: kv_block_managers,
        BlockType.dKV: dkv_block_managers,
        BlockType.Out: out_block_managers,
        BlockType.dOut: dout_block_managers,
        BlockType.LSE: lse_block_managers,
    }
    _BUFFER_TYPE_TO_MANAGERS = {
        BufferType.BUFFER_Q: q_block_managers,
        BufferType.BUFFER_dQ: dq_block_managers,
        BufferType.BUFFER_KV: kv_block_managers,
        BufferType.BUFFER_dKV: dkv_block_managers,
        BufferType.BUFFER_OUT: out_block_managers,
        BufferType.BUFFER_dOUT: dout_block_managers,
        BufferType.BUFFER_LSE: lse_block_managers,
    }

    def _get_data_block_type(data_id: int, is_input: bool):
        if is_input:
            block_type = workload.block_mapping.input_id_to_meta[data_id].type
        else:
            block_type = workload.block_mapping.output_id_to_meta[data_id].type
        return block_type

    def _get_buffer_manager(d: Tuple[int, int], data_id: int, is_input: bool):
        block_type = _get_data_block_type(data_id, is_input)
        if block_type not in _BUFFER_BLOCK_TYPE_TO_BUFFER_TYPE:
            raise ValueError(f"Unknown buffer block type: {block_type}")
        buffer_manager = _BLOCK_TYPE_TO_MANAGERS[block_type][d]
        buffer_type = _BUFFER_BLOCK_TYPE_TO_BUFFER_TYPE[block_type]
        return buffer_manager, buffer_type

    def _get_local_buffer(d: Tuple[int, int], data_id: int, is_input: bool):
        # first check if it is in local input/output buffers
        if is_input:
            src_d = input_to_device_map[data_id]
            block_type = workload.block_mapping.input_id_to_meta[data_id].type
            block_index = workload.block_mapping.input_id_to_buffer_index[
                src_d
            ][data_id]
            block_n_tokens = workload.block_mapping.input_id_to_meta[
                data_id
            ].n_tokens
        else:
            src_d = output_to_device_map[data_id]
            block_type = workload.block_mapping.output_id_to_meta[data_id].type
            block_index = workload.block_mapping.output_id_to_buffer_index[
                src_d
            ][data_id]
            block_n_tokens = workload.block_mapping.output_id_to_meta[
                data_id
            ].n_tokens
        assert src_d == d
        assert (
            block_type in _LOCAL_BLOCK_TYPE_TO_BUFFER_TYPE
        ), f"Unknown block type: {block_type} in local buffer."
        buffer_type = _LOCAL_BLOCK_TYPE_TO_BUFFER_TYPE[block_type]
        return (
            BufferBlock(
                buffer_type,
                block_index,
                block_n_tokens,
            ),
            -1,
        )

    def _get_single_buffer(
        d: Tuple[int, int],
        data_id: int,
        is_input: bool,
        return_stage_id: bool = False,
        allow_local: bool = False,
        expect_exists: bool = True,
    ):
        buffer_manager, buffer_type = _get_buffer_manager(d, data_id, is_input)
        if buffer_manager.has_block(data_id, is_input, d):
            block_stage_and_replica_ids = (
                buffer_manager.get_block_all_versions(data_id, is_input, d)
            )
            block_ids, stage_ids, _ = zip(*block_stage_and_replica_ids)
            if is_input:
                n_tokens = workload.block_mapping.input_id_to_meta[
                    data_id
                ].n_tokens
            else:
                n_tokens = workload.block_mapping.output_id_to_meta[
                    data_id
                ].n_tokens
            assert (
                len(block_ids) == 1
            ), "Multiple blocks found for data_id: {}".format(data_id)
            if return_stage_id:
                return (
                    BufferBlock(buffer_type, block_ids[0], n_tokens),
                    stage_ids[0],
                )
            else:
                return BufferBlock(buffer_type, block_ids[0], n_tokens)
        if expect_exists and allow_local:
            local_buffer, stage_id = _get_local_buffer(d, data_id, is_input)
            if return_stage_id:
                return local_buffer, stage_id
            else:
                return local_buffer
        if expect_exists:
            raise ValueError(
                f"Block not found for data_id {data_id} on device {d}"
            )
        if return_stage_id:
            return None, None
        else:
            return None

    def _get_buffer_all_versions(
        d: Tuple[int, int],
        data_id: int,
        is_input: bool,
        expect_exists: bool = True,
    ):
        buffer_manager, buffer_type = _get_buffer_manager(d, data_id, is_input)
        if is_input:
            n_tokens = workload.block_mapping.input_id_to_meta[
                data_id
            ].n_tokens
        else:
            n_tokens = workload.block_mapping.output_id_to_meta[
                data_id
            ].n_tokens
        if buffer_manager.has_block(data_id, is_input):
            blocks = buffer_manager.get_block_all_versions(data_id, is_input)
            block_ids, stage_ids, creation_devices, replica_ids = zip(*blocks)
            buffer_blocks = [
                BufferBlock(
                    buffer_type,
                    block_id,
                    n_tokens,
                )
                for block_id in block_ids
            ]
            return buffer_blocks, stage_ids, creation_devices, replica_ids
        else:
            if expect_exists:
                raise ValueError(
                    f"Block not found for data_id {data_id} on device {d}"
                )
            return None, None, None, None

    def _create_buffer(
        d: Tuple[int, int],
        data_id: int,
        is_input: bool,
        stage_id: int,
        creation_device: Optional[Tuple[int, int]] = None,
        replica_id: Optional[int] = None,
    ):
        buffer_manager, buffer_type = _get_buffer_manager(d, data_id, is_input)
        if is_input:
            n_tokens = workload.block_mapping.input_id_to_meta[
                data_id
            ].n_tokens
        else:
            n_tokens = workload.block_mapping.output_id_to_meta[
                data_id
            ].n_tokens
        if creation_device is None:
            creation_device = d
        if replica_id is None:
            replica_id = 0
        buffer_index = buffer_manager.alloc_block(
            data_id,
            is_input,
            creation_device,
            stage_id=stage_id,
            replica_id=replica_id,
        )
        return BufferBlock(
            buffer_type,
            buffer_index,
            n_tokens,
        )

    def _free_buffer(d: Tuple[int, int], buffer_block: BufferBlock):
        if buffer_block.buffer_type not in _BUFFER_TYPE_TO_MANAGERS:
            raise ValueError(
                f"Unknown buffer type: {buffer_block.buffer_type}"
            )
        buffer_manager = _BUFFER_TYPE_TO_MANAGERS[buffer_block.buffer_type][d]
        buffer_manager.free_block_by_idx(buffer_block.index)

    # run a liveliness analysis to determine when to free the blocks
    per_device_input_live_stage = defaultdict(
        lambda: defaultdict(set)
    )  # d -> Dict[input_id -> Set[stage_id]]
    for stage_id, stage in enumerate(workload_per_stage):
        for d, workload_ids in stage.items():
            for workload_id in workload_ids:
                for input_id in workload.work_unit_input_map[workload_id]:
                    per_device_input_live_stage[d][input_id].add(stage_id)
    for stage_id, stage_comm in enumerate(comms_per_stage):
        for dst_d, comms in stage_comm.items():
            for input_id, src_d in comms:
                per_device_input_live_stage[dst_d][input_id].add(stage_id)
                per_device_input_live_stage[src_d][input_id].add(stage_id - 1)

    if logging_enabled:
        _logger_debug("Input live stages per device:")
        for d, input_live_stage_map in per_device_input_live_stage.items():
            _logger_debug(f"\tDevice {d}: {input_live_stage_map}")
    per_device_input_live_range = {}
    for d, input_liveliness_map in per_device_input_live_stage.items():
        per_device_input_live_range[d] = {}
        for input_id, live_stages in input_liveliness_map.items():
            per_device_input_live_range[d][input_id] = (
                min(live_stages),
                max(live_stages),
            )
    if logging_enabled:
        _logger_debug("N unique inputs: {}".format(len(input_to_device_map)))
        _logger_debug("Input live ranges per device:")
        for d, input_live_range_map in per_device_input_live_range.items():
            _logger_debug(f"\tDevice {d}: {input_live_range_map}")

    # start scheduling
    # for each stage, generate
    # 1. communication instructions if some input is not available locally
    # 2. computation instructions, filling in the input and output args
    per_device_instructions = defaultdict(list)  # d -> List[List[Instr]]
    pending_comm_keys = defaultdict(set)  # d -> set[str]

    def _inject_barrier():
        for d in all_devices:
            per_device_instructions[d].append(BarrierInstr())

    def _copy_input_to_local_buffer():
        # first get all locally used inputs across all stages
        locally_used_inputs = defaultdict(set)
        for workload_dict in workload_per_stage:
            for d, workload_ids in workload_dict.items():
                for workload_id in workload_ids:
                    for input_id in workload.work_unit_input_map[workload_id]:
                        if input_to_device_map[input_id] == d:
                            locally_used_inputs[d].add(input_id)
        # then copy them to local buffers
        for d, input_ids in locally_used_inputs.items():
            memcpy_block_pairs = []
            # sorting makes sure that out, lse and dout share the same
            # buffer id, which is important since we only use a single
            # out block table for all three.
            # TODO: reconsider this design later
            for input_id in sorted(input_ids):
                src_block, _ = _get_local_buffer(d, input_id, is_input=True)
                dst_block = _create_buffer(
                    d, input_id, is_input=True, stage_id=-1
                )
                memcpy_block_pairs.append((src_block, dst_block))
            per_device_instructions[d].append(MemcpyInstr(memcpy_block_pairs))

    def _gen_input_comm_instrs_for_stage(
        per_stage_instructions: Dict[Tuple[int, int], List], stage_id: int
    ):
        _logger_debug(
            "Calling  _gen_input_comm_instrs_for_stage for stage {}".format(
                stage_id
            )
        )
        required_comm_per_device = defaultdict(list)
        if (
            stage_id == len(workload_per_stage) - 1
            and len(workload_per_stage) > 1
        ):
            return required_comm_per_device

        comm_stage_id = (stage_id + 1) if len(workload_per_stage) != 1 else 0

        current_internode_comm_workload = defaultdict(int)
        comm_op_sizes = defaultdict(int)

        scheduled_comms = comms_per_stage[comm_stage_id]
        for dst_d, comms in scheduled_comms.items():
            _logger_debug(
                "Comm stage: {}, Scheduled comms for device {}: {}".format(
                    comm_stage_id, dst_d, comms
                )
            )
            for input_id, src_d in comms:
                src_block = _get_single_buffer(
                    src_d, input_id, is_input=True, allow_local=True
                )
                dst_block = _create_buffer(
                    dst_d, input_id, is_input=True, stage_id=comm_stage_id
                )
                required_comm_per_device[dst_d].append(
                    (src_d, src_block, dst_block)
                )
                if src_d[0] != dst_d[0]:
                    current_internode_comm_workload[
                        src_d
                    ] += workload.input_sizes[input_id]
                comm_op_sizes[(src_d, dst_d, src_block, dst_block)] = (
                    workload.input_sizes[input_id]
                )

        second_phase_comms = defaultdict(
            list
        )  # d -> List[(peer_d, src_buffer, dst_buffer)]
        for dst_d, comms in second_phase_comm_per_stage[comm_stage_id].items():
            _logger_debug(
                "Comm stage: {}, Second phase comms for device {}: {}".format(
                    comm_stage_id, dst_d, comms
                )
            )
            for input_id, src_d in comms:
                src_block = _get_single_buffer(
                    src_d, input_id, is_input=True, allow_local=True
                )
                dst_block = _create_buffer(
                    dst_d, input_id, is_input=True, stage_id=comm_stage_id
                )
                second_phase_comms[dst_d].append((src_d, src_block, dst_block))
                if src_d[0] != dst_d[0]:
                    current_internode_comm_workload[
                        src_d
                    ] += workload.input_sizes[input_id]
                comm_op_sizes[(src_d, dst_d, src_block, dst_block)] = (
                    workload.input_sizes[input_id]
                )

        for d, comm_size in current_internode_comm_workload.items():
            _logger_debug(
                f"Stage {stage_id}, Device {d} total internode comm workload: {comm_size}"
            )

        # generate first phase communication instructions
        comm_ops_per_device = defaultdict(list)  # d -> List[CommOp]
        for curr_d, comm_ops in required_comm_per_device.items():
            for (
                peer_d,
                src_buffer,
                dst_buffer,
            ) in comm_ops:
                # send
                comm_ops_per_device[peer_d].append(
                    CommOp(CommType.SEND, curr_d, src_buffer)
                )
                # recv
                comm_ops_per_device[curr_d].append(
                    CommOp(CommType.RECV, peer_d, dst_buffer)
                )
        # inject communication instructions
        if comm_ops_per_device:
            comm_stage_id = (
                stage_id + 1 if not len(workload_per_stage) == 1 else 0
            )
            comm_key = f"S{comm_stage_id}"
            for d in all_devices:
                if d in comm_ops_per_device:
                    per_stage_instructions[d].append(
                        CommLaunchInstr(comm_key, comm_ops_per_device[d])
                    )
                    pending_comm_keys[d].add(comm_key)
            if second_phase_comms:
                # generate second phase communication instructions
                comm_ops_per_device = defaultdict(list)
                for curr_d, comm_tuples in second_phase_comms.items():
                    for (
                        peer_d,
                        src_buffer,
                        dst_buffer,
                    ) in comm_tuples:
                        # _logger_debug(
                        #     "Second phase comm: {}:{} -> {}:{}".format(
                        #         peer_d, src_buffer, curr_d, dst_buffer
                        #     )
                        # )
                        # send
                        comm_ops_per_device[peer_d].append(
                            CommOp(CommType.SEND, curr_d, src_buffer)
                        )
                        # recv
                        comm_ops_per_device[curr_d].append(
                            CommOp(CommType.RECV, peer_d, dst_buffer)
                        )
                # inject communication instructions
                second_phase_comm_key = f"S{comm_stage_id}P2"
                for d in all_devices:
                    if d in comm_ops_per_device:
                        # first wait for the first phase to finish on another stream
                        if comm_key in pending_comm_keys[d]:
                            per_stage_instructions[d].append(
                                CommWaitInstr(comm_key, stream="second_phase")
                            )
                            pending_comm_keys[d].remove(comm_key)
                        per_stage_instructions[d].append(
                            CommLaunchInstr(
                                second_phase_comm_key,
                                comm_ops_per_device[d],
                                stream="second_phase",
                            )
                        )
                        pending_comm_keys[d].add(second_phase_comm_key)

    def _wait_for_input_comm(stage_id: int):
        if stage_id == 0 and len(workload_per_stage) > 1:
            return
        for d in all_devices:
            phase_one_key = f"S{stage_id}"
            phase_two_key = f"S{stage_id}P2"
            if phase_one_key in pending_comm_keys[d]:
                per_stage_instructions[d].append(CommWaitInstr(phase_one_key))
                pending_comm_keys[d].remove(phase_one_key)
            if phase_two_key in pending_comm_keys[d]:
                per_stage_instructions[d].append(CommWaitInstr(phase_two_key))
                pending_comm_keys[d].remove(phase_two_key)

    def _gen_attn_fw_instr_for_stage(
        per_stage_instructions: Dict[Tuple[int, int], List],
        stage_id: int,
        stage: Dict[Tuple[int, int], List[int]],
    ):
        _logger_debug(
            "Calling  _gen_attn_fw_instr_for_stage for stage {}".format(
                stage_id
            )
        )
        # calculate q_seqlens, kv_seqlens
        # we only have one block per each q, so q seqlen is the same as block
        # size
        q_seqlens_per_device = {}
        kv_seqlens_per_device = {}
        attn_mask_per_device = {}
        q_block_tables_per_device = {}
        kv_block_tables_per_device = {}
        out_block_tables_per_device = {}
        lse_block_tables_per_device = {}

        # get unique q blocks for each device
        unique_q_blocks_per_device = defaultdict(set)
        unique_out_lse_blocks_per_device = defaultdict(set)
        for d, workload_ids in stage.items():
            for workload_id in workload_ids:
                work_meta: ComputationBlockMeta = (
                    workload.block_mapping.work_id_to_meta[workload_id]
                )
                unique_q_blocks_per_device[d].add(work_meta.q_id)
                unique_out_lse_blocks_per_device[d].add(
                    (work_meta.out_id, work_meta.lse_id)
                )

        for d, curr_stage_qs in unique_q_blocks_per_device.items():
            curr_stage_qs_in_order = sorted(list(curr_stage_qs))
            q_seqlens_per_device[d] = [
                workload.block_mapping.input_id_to_meta[q_id].block_size
                for q_id in curr_stage_qs_in_order
            ]
            q_id_to_kv_ids = defaultdict(list)
            q_id_to_attn_masks = defaultdict(list)
            q_id_to_out_id = {}
            q_id_to_lse_id = {}
            for workload_id in stage[d]:
                work_meta: ComputationBlockMeta = (
                    workload.block_mapping.work_id_to_meta[workload_id]
                )
                q_id = work_meta.q_id
                kv_id = work_meta.kv_id
                out_id = work_meta.out_id
                lse_id = work_meta.lse_id
                q_id_to_kv_ids[q_id].append(kv_id)
                q_id_to_attn_masks[q_id].append(work_meta.local_attn_mask)
                assert work_meta.local_attn_mask.dtype == torch.int32
                if q_id not in q_id_to_out_id:
                    q_id_to_out_id[q_id] = out_id
                else:
                    assert q_id_to_out_id[q_id] == out_id
                if q_id not in q_id_to_lse_id:
                    q_id_to_lse_id[q_id] = lse_id
                else:
                    assert q_id_to_lse_id[q_id] == lse_id
            # kv seqlens and block tables
            q_block_table: List[BufferBlock] = []
            kv_block_table: List[BufferBlock] = []
            out_block_table: List[BufferBlock] = []
            lse_block_table: List[BufferBlock] = []
            kv_seqlens = []
            attn_mask = None
            for q_id in curr_stage_qs_in_order:
                kv_ids = q_id_to_kv_ids[q_id]
                local_attn_masks = q_id_to_attn_masks[q_id]
                # sort by kv block id
                kv_ids, local_attn_masks = zip(
                    *sorted(
                        zip(kv_ids, local_attn_masks),
                        key=lambda x: workload.block_mapping.input_id_to_meta[
                            x[0]
                        ].block_id,
                    )
                )
                q_seqlen = workload.block_mapping.input_id_to_meta[
                    q_id
                ].block_size
                curr_q_kv_seqlens = [
                    workload.block_mapping.input_id_to_meta[kv_id].block_size
                    for kv_id in kv_ids
                ]
                kv_seqlens.append(sum(curr_q_kv_seqlens))
                assert len(local_attn_masks) == len(kv_ids)

                def _concat_attn_mask(concated_attn_mask, local_attn_masks):
                    cumu_kv_seqlen = torch.zeros(
                        q_seqlen, dtype=torch.int32, device="cpu"
                    )
                    for mask_id, mask in enumerate(local_attn_masks):
                        assert (
                            mask.dtype == torch.int32
                        ), "Expect int32 attn mask"
                        # start_time = time.time()
                        start = mask[0]
                        end = mask[1]
                        valid_range = end > start

                        new_start = start + cumu_kv_seqlen
                        new_end = end + cumu_kv_seqlen

                        assert torch.all(
                            (concated_attn_mask[0] == -1)
                            | (new_start == concated_attn_mask[1])
                            | (~valid_range)
                        ), "Discontinuous attn mask"

                        indices_ = concated_attn_mask[0] == -1
                        concated_attn_mask[0][valid_range & indices_] = (
                            new_start[valid_range & indices_]
                        )
                        concated_attn_mask[1][valid_range] = new_end[
                            valid_range
                        ]

                        cumu_kv_seqlen += curr_q_kv_seqlens[mask_id]
                        if mask_id != len(local_attn_masks) - 1:
                            assert (
                                curr_q_kv_seqlens[mask_id]
                                == workload.block_mapping.input_id_to_meta[
                                    kv_ids[mask_id]
                                ].block_size
                            )

                    invalid_range = concated_attn_mask[0] == -1
                    concated_attn_mask[0][invalid_range] = 0
                    concated_attn_mask[1][invalid_range] = 0

                    # end_time = time.time()
                    # print(
                    #     f"Operation time: {end_time - start_time} seconds",
                    #     flush=True,
                    # )

                if local_attn_masks[0].dim() == 2:
                    # 2d attn mask, should have a single range for each
                    # q_index
                    concatenated_attn_mask = [
                        torch.full(
                            (q_seqlen,), -1, dtype=torch.int32, device="cpu"
                        )
                        for _ in range(2)
                    ]
                    start_time = time.time()
                    _concat_attn_mask(concatenated_attn_mask, local_attn_masks)
                    end_time = time.time()
                    # print(f"Concatenation operation took {end_time - start_time} seconds", flush=True)
                    if attn_mask is None:
                        attn_mask = concatenated_attn_mask
                    else:
                        for i in range(2):
                            attn_mask[i] = torch.cat(
                                [attn_mask[i], concatenated_attn_mask[i]]
                            )
                else:
                    # 3d attn mask, could have two ranges for each q_index
                    concatenated_attn_mask = [
                        [
                            torch.full(
                                (q_seqlen,),
                                -1,
                                dtype=torch.int32,
                                device="cpu",
                            )
                            for _ in range(2)
                        ]
                        for _ in range(2)
                    ]
                    start_time = time.time()
                    for i in range(2):
                        _concat_attn_mask(
                            concatenated_attn_mask[i],
                            [mask[i] for mask in local_attn_masks],
                        )
                    end_time = time.time()
                    # print(f"Concatenation operation took {end_time - start_time} seconds", flush=True)
                    if attn_mask is None:
                        attn_mask = concatenated_attn_mask
                    else:
                        for i in range(2):
                            for j in range(2):
                                attn_mask[i][j] = torch.cat(
                                    [
                                        attn_mask[i][j],
                                        concatenated_attn_mask[i][j],
                                    ]
                                )

                out_id = q_id_to_out_id[q_id]
                lse_id = q_id_to_lse_id[q_id]
                q_block_table.append(
                    [_get_single_buffer(d, q_id, is_input=True)]
                )
                kv_block_table.append(
                    [
                        _get_single_buffer(d, kv_id, is_input=True)
                        for kv_id in kv_ids
                    ]
                )
                # first see if the output buffer is already created
                out_buffer, out_creation_stage = _get_single_buffer(
                    d,
                    out_id,
                    is_input=False,
                    return_stage_id=True,
                    expect_exists=False,
                )
                if out_buffer is None or out_creation_stage < stage_id:
                    # if out_creation_stage < stage_id, it means that
                    # the output buffer is created in a previous stage
                    # we need to create a new buffer for the current stage
                    # and perform reduciton at the end of the stage
                    out_buffer = _create_buffer(
                        d, out_id, is_input=False, stage_id=stage_id
                    )
                out_block_table.append([out_buffer])
                lse_buffer, lse_creation_stage = _get_single_buffer(
                    d,
                    lse_id,
                    is_input=False,
                    return_stage_id=True,
                    expect_exists=False,
                )
                if lse_buffer is None or lse_creation_stage < stage_id:
                    lse_buffer = _create_buffer(
                        d, lse_id, is_input=False, stage_id=stage_id
                    )
                lse_block_table.append([lse_buffer])
            # concat
            kv_seqlens_per_device[d] = kv_seqlens
            if isinstance(attn_mask[0], list):
                # 3d attn mask, stack twice
                attn_mask_per_device[d] = torch.stack(
                    [
                        torch.stack(attn_mask[0], dim=0),
                        torch.stack(attn_mask[1], dim=0),
                    ],
                    dim=0,
                )
            else:
                attn_mask_per_device[d] = torch.stack(attn_mask, dim=0)
            q_block_tables_per_device[d] = q_block_table
            kv_block_tables_per_device[d] = kv_block_table
            out_block_tables_per_device[d] = out_block_table
            lse_block_tables_per_device[d] = lse_block_table
        # add instructions
        for d in all_devices:
            if d not in q_seqlens_per_device:
                continue
            max_seqlens_q = max(q_seqlens_per_device[d])
            max_seqlens_kv = max(kv_seqlens_per_device[d])
            per_stage_instructions[d].append(
                AttnInstr(
                    stage_id,
                    q_seqlens_per_device[d],
                    kv_seqlens_per_device[d],
                    max_seqlens_q,
                    max_seqlens_kv,
                    attn_mask_per_device[d],
                    q_block_tables_per_device[d],
                    kv_block_tables_per_device[d],
                    out_block_tables_per_device[d],
                    lse_block_tables_per_device[d],
                )
            )
        # now see if we need to reduce the output buffers
        for d in all_devices:
            reduction_ops = []
            for out_id, lse_id in unique_out_lse_blocks_per_device[d]:
                out_buffers, out_creation_stages, _, _ = (
                    _get_buffer_all_versions(d, out_id, is_input=False)
                )
                lse_buffers, lse_creation_stages, _, _ = (
                    _get_buffer_all_versions(d, lse_id, is_input=False)
                )
                if len(out_buffers) == 1:
                    continue
                # sort by creation stage
                sorted_out_buffers, _ = zip(
                    *sorted(
                        zip(out_buffers, out_creation_stages),
                        key=lambda x: x[1],
                    )
                )
                sorted_lse_buffers, _ = zip(
                    *sorted(
                        zip(lse_buffers, lse_creation_stages),
                        key=lambda x: x[1],
                    )
                )
                assert len(sorted_out_buffers) == len(sorted_lse_buffers) == 2
                red_op = ReductionOp(
                    [
                        (sorted_out_buffers[0], sorted_lse_buffers[0]),
                        (sorted_out_buffers[1], sorted_lse_buffers[1]),
                    ],
                    (sorted_out_buffers[0], sorted_lse_buffers[0]),
                )
                reduction_ops.append(red_op)
                # free the newer buffers
                _free_buffer(d, sorted_out_buffers[1])
                _free_buffer(d, sorted_lse_buffers[1])
            if reduction_ops:
                per_stage_instructions[d].append(
                    AttnReductionInstr(reduction_ops, output_is_unpadded=0)
                )

    def _gen_attn_bw_instr_for_stage(
        per_stage_instructions: Dict[Tuple[int, int], List],
        stage_id: int,
        stage: Dict[Tuple[int, int], List[int]],
    ):
        _logger_debug(
            "Calling  _gen_attn_bw_instr_for_stage for stage {}".format(
                stage_id
            )
        )
        # calculate q_seqlens, kv_seqlens, attn_mask
        # we only have one block per each q, so q seqlen is the same as block
        # size
        q_seqlens_per_device = {}
        kv_seqlens_per_device = {}
        attn_mask_per_device = {}
        q_block_tables_per_device = {}
        kv_block_tables_per_device = {}
        out_block_tables_per_device = {}
        dq_block_tables_per_device = {}
        dkv_block_tables_per_device = {}

        # get unique q blocks for each device
        unique_q_blocks_per_device = defaultdict(set)
        unique_dq_blocks_per_device = defaultdict(set)
        unique_dkv_blocks_per_device = defaultdict(set)
        for d, workload_ids in stage.items():
            for workload_id in workload_ids:
                work_meta: ComputationBlockMeta = (
                    workload.block_mapping.work_id_to_meta[workload_id]
                )
                unique_q_blocks_per_device[d].add(work_meta.q_id)
                unique_dq_blocks_per_device[d].add(work_meta.dq_id)
                unique_dkv_blocks_per_device[d].add(work_meta.dkv_id)

        for d, curr_stage_qs in unique_q_blocks_per_device.items():
            curr_stage_qs_in_order = sorted(list(curr_stage_qs))
            q_seqlens_per_device[d] = [
                workload.block_mapping.input_id_to_meta[q_id].block_size
                for q_id in curr_stage_qs_in_order
            ]
            q_id_to_kv_ids = defaultdict(list)
            q_id_to_attn_masks = defaultdict(list)
            q_id_to_out_id = {}
            q_id_to_dout_id = {}
            q_id_to_lse_id = {}
            q_id_to_dq_id = {}
            q_id_to_dkv_ids = defaultdict(list)
            for workload_id in stage[d]:
                work_meta: ComputationBlockMeta = (
                    workload.block_mapping.work_id_to_meta[workload_id]
                )
                q_id = work_meta.q_id
                kv_id = work_meta.kv_id
                out_id = work_meta.out_id
                lse_id = work_meta.lse_id
                dq_id = work_meta.dq_id
                dkv_id = work_meta.dkv_id
                dout_id = work_meta.dout_id
                q_id_to_kv_ids[q_id].append(kv_id)
                q_id_to_dkv_ids[q_id].append(dkv_id)
                q_id_to_attn_masks[q_id].append(work_meta.local_attn_mask)
                if q_id not in q_id_to_out_id:
                    q_id_to_out_id[q_id] = out_id
                else:
                    assert q_id_to_out_id[q_id] == out_id
                if q_id not in q_id_to_lse_id:
                    q_id_to_lse_id[q_id] = lse_id
                else:
                    assert q_id_to_lse_id[q_id] == lse_id
                if q_id not in q_id_to_dout_id:
                    q_id_to_dout_id[q_id] = dout_id
                else:
                    assert q_id_to_dout_id[q_id] == dout_id
                if q_id not in q_id_to_dq_id:
                    q_id_to_dq_id[q_id] = dq_id
                else:
                    assert q_id_to_dq_id[q_id] == dq_id
            # kv seqlens and block tables
            q_block_table: List[List[BufferBlock]] = []
            kv_block_table: List[List[BufferBlock]] = []
            out_block_table: List[List[BufferBlock]] = []
            dq_block_table: List[List[BufferBlock]] = []
            dkv_block_table: List[List[BufferBlock]] = []
            kv_seqlens = []
            attn_mask = None
            dkv_occurence = defaultdict(int)
            for q_id in curr_stage_qs_in_order:
                kv_ids = q_id_to_kv_ids[q_id]
                local_attn_masks = q_id_to_attn_masks[q_id]
                # sort by kv block id
                kv_ids, local_attn_masks = zip(
                    *sorted(
                        zip(kv_ids, local_attn_masks),
                        key=lambda x: workload.block_mapping.input_id_to_meta[
                            x[0]
                        ].block_id,
                    )
                )
                curr_q_kv_seqlens = [
                    workload.block_mapping.input_id_to_meta[kv_id].block_size
                    for kv_id in kv_ids
                ]
                kv_seqlens.append(sum(curr_q_kv_seqlens))
                assert len(local_attn_masks) == len(kv_ids)

                q_seqlen = workload.block_mapping.input_id_to_meta[
                    q_id
                ].block_size

                def _concat_attn_mask(concated_attn_mask, local_attn_masks):
                    cumu_kv_seqlen = torch.zeros(
                        q_seqlen, dtype=torch.int32, device="cpu"
                    )
                    for mask_id, mask in enumerate(local_attn_masks):
                        assert (
                            mask.dtype == torch.int32
                        ), "Expect int32 attn mask"
                        # start_time = time.time()
                        start = mask[0]
                        end = mask[1]
                        valid_range = end > start

                        new_start = start + cumu_kv_seqlen
                        new_end = end + cumu_kv_seqlen

                        assert torch.all(
                            (concated_attn_mask[0] == -1)
                            | (new_start == concated_attn_mask[1])
                            | (~valid_range)
                        ), "Discontinuous attn mask"

                        indices_ = concated_attn_mask[0] == -1
                        concated_attn_mask[0][valid_range & indices_] = (
                            new_start[valid_range & indices_]
                        )
                        concated_attn_mask[1][valid_range] = new_end[
                            valid_range
                        ]

                        cumu_kv_seqlen += curr_q_kv_seqlens[mask_id]
                        if mask_id != len(local_attn_masks) - 1:
                            assert (
                                curr_q_kv_seqlens[mask_id]
                                == workload.block_mapping.input_id_to_meta[
                                    kv_ids[mask_id]
                                ].block_size
                            )

                    invalid_range = concated_attn_mask[0] == -1
                    concated_attn_mask[0][invalid_range] = 0
                    concated_attn_mask[1][invalid_range] = 0

                    # end_time = time.time()
                    # print(
                    #     f"Operation time: {end_time - start_time} seconds",
                    #     flush=True,
                    # )

                if local_attn_masks[0].dim() == 2:
                    # 2d attn mask, should have a single range for each
                    # q_index
                    concatenated_attn_mask = [
                        torch.full(
                            (q_seqlen,), -1, dtype=torch.int32, device="cpu"
                        )
                        for _ in range(2)
                    ]
                    start_time = time.time()
                    _concat_attn_mask(concatenated_attn_mask, local_attn_masks)
                    end_time = time.time()
                    # print(f"Concatenation operation took {end_time - start_time} seconds", flush=True)
                    if attn_mask is None:
                        attn_mask = concatenated_attn_mask
                    else:
                        for i in range(2):
                            attn_mask[i] = torch.cat(
                                [attn_mask[i], concatenated_attn_mask[i]]
                            )
                else:
                    # 3d attn mask, could have two ranges for each q_index
                    concatenated_attn_mask = [
                        [
                            torch.full(
                                (q_seqlen,),
                                -1,
                                dtype=torch.int32,
                                device="cpu",
                            )
                            for _ in range(2)
                        ]
                        for _ in range(2)
                    ]
                    start_time = time.time()
                    for i in range(2):
                        _concat_attn_mask(
                            concatenated_attn_mask[i],
                            [mask[i] for mask in local_attn_masks],
                        )
                    end_time = time.time()
                    # print(f"Concatenation operation took {end_time - start_time} seconds", flush=True)
                    if attn_mask is None:
                        attn_mask = concatenated_attn_mask
                    else:
                        for i in range(2):
                            for j in range(2):
                                attn_mask[i][j] = torch.cat(
                                    [
                                        attn_mask[i][j],
                                        concatenated_attn_mask[i][j],
                                    ]
                                )
                dout_id = q_id_to_dout_id[q_id]
                out_id = q_id_to_out_id[q_id]
                lse_id = q_id_to_lse_id[q_id]
                dq_id = q_id_to_dq_id[q_id]
                dkv_ids = q_id_to_dkv_ids[q_id]
                q_block_table.append(
                    [_get_single_buffer(d, q_id, is_input=True)]
                )
                kv_block_table.append(
                    [
                        _get_single_buffer(d, kv_id, is_input=True)
                        for kv_id in kv_ids
                    ]
                )
                out_block = _get_single_buffer(d, out_id, is_input=True)
                lse_block = _get_single_buffer(d, lse_id, is_input=True)
                dout_block = _get_single_buffer(d, dout_id, is_input=True)
                out_block_table.append([out_block])
                assert (
                    out_block.index == lse_block.index == dout_block.index
                ), f"block id mismatch (Out: {out_block}, LSE: {lse_block}, Dout: {dout_block})"
                # handle dq and dkv
                dq_buffer = _create_buffer(
                    d, dq_id, is_input=False, stage_id=stage_id
                )
                dq_block_table.append([dq_buffer])
                dkv_buffers = []
                for dkv_id in dkv_ids:
                    occurence_in_current_stage = dkv_occurence[dkv_id]
                    dkv_buffer = _create_buffer(
                        d,
                        dkv_id,
                        is_input=False,
                        stage_id=stage_id,
                        replica_id=occurence_in_current_stage,
                    )
                    dkv_buffers.append(dkv_buffer)
                    dkv_occurence[dkv_id] += 1
                dkv_block_table.append(dkv_buffers)

            kv_seqlens_per_device[d] = kv_seqlens
            q_block_tables_per_device[d] = q_block_table
            kv_block_tables_per_device[d] = kv_block_table
            if isinstance(attn_mask[0], list):
                # 3d attn mask, stack twice
                attn_mask_per_device[d] = torch.stack(
                    [
                        torch.stack(attn_mask[0], dim=0),
                        torch.stack(attn_mask[1], dim=0),
                    ],
                    dim=0,
                )
            else:
                attn_mask_per_device[d] = torch.stack(attn_mask, dim=0)
            out_block_tables_per_device[d] = out_block_table
            dq_block_tables_per_device[d] = dq_block_table
            dkv_block_tables_per_device[d] = dkv_block_table
        # add instructions
        for d in all_devices:
            if d not in q_seqlens_per_device:
                continue
            max_seqlens_q = max(q_seqlens_per_device[d])
            max_seqlens_kv = max(kv_seqlens_per_device[d])
            per_stage_instructions[d].append(
                AttnBackwardInstr(
                    stage_id,
                    q_seqlens_per_device[d],
                    kv_seqlens_per_device[d],
                    max_seqlens_q,
                    max_seqlens_kv,
                    attn_mask_per_device[d],
                    q_block_tables_per_device[d],
                    kv_block_tables_per_device[d],
                    out_block_tables_per_device[d],
                    dq_block_tables_per_device[d],
                    dkv_block_tables_per_device[d],
                )
            )
        # now see if we need to reduce the output buffers
        for d in all_devices:
            sum_ops = []
            for dq_id in unique_dq_blocks_per_device[d]:
                dq_buffers, dq_creation_stages, _, _ = (
                    _get_buffer_all_versions(d, dq_id, is_input=False)
                )
                if len(dq_buffers) == 1:
                    continue
                # sort by creation stage
                sorted_dq_buffers, _ = zip(
                    *sorted(
                        zip(dq_buffers, dq_creation_stages),
                        key=lambda x: x[1],
                    )
                )
                assert len(sorted_dq_buffers) == 2
                sum_op = SumOp(
                    [sorted_dq_buffers[0], sorted_dq_buffers[1]],
                    sorted_dq_buffers[0],
                )
                sum_ops.append(sum_op)
                # free the newer buffers
                _free_buffer(d, sorted_dq_buffers[1])
            for dkv_id in unique_dkv_blocks_per_device[d]:
                dkv_buffers, dkv_creation_stages, _, dkv_replica_ids = (
                    _get_buffer_all_versions(d, dkv_id, is_input=False)
                )
                if len(dkv_buffers) == 1:
                    continue
                # sort by creation stage and replica id
                sorted_dkv_buffers, _, _ = zip(
                    *sorted(
                        zip(dkv_buffers, dkv_creation_stages, dkv_replica_ids),
                        key=lambda x: (x[1], x[2]),
                    )
                )
                sum_op = SumOp(sorted_dkv_buffers, sorted_dkv_buffers[0])
                sum_ops.append(sum_op)
                # free the newer buffers
                for dkv_buffer in sorted_dkv_buffers[1:]:
                    _free_buffer(d, dkv_buffer)
            if sum_ops:
                per_stage_instructions[d].append(
                    SumInstr(sum_ops, output_is_unpadded=0)
                )

    def _gen_output_comm_instrs():
        # for the same device, if different workloads have the same output id
        # their output will be already reduced, so we only count the unique
        # output ids per device
        unique_output_ids_per_device = defaultdict(set)
        for stage_id, stage in enumerate(workload_per_stage):
            for d, workload_ids in stage.items():
                for workload_id in workload_ids:
                    for output_id in workload.work_unit_output_map[
                        workload_id
                    ]:
                        unique_output_ids_per_device[d].add(output_id)
        if logging_enabled:
            _logger_debug("Unique output ids per device:")
            for d, output_ids in unique_output_ids_per_device.items():
                _logger_debug(f"\tDevice {d}: {output_ids}")

        # now, generate communication instrs to copy the final output buffers
        # to the output location
        per_device_comm_ops = defaultdict(list)  # d -> List[CommOp]
        for (
            d,
            output_ids,
        ) in unique_output_ids_per_device.items():
            # sorting here is important since we want to make sure that out
            # and lse share the same buffer id.
            # TODO: fix later, needs kernel modifications
            for output_id in sorted(output_ids):
                src_buffer = _get_single_buffer(d, output_id, is_input=False)
                assert src_buffer is not None, "output buffer not found"
                dst_d = output_to_device_map[output_id]
                if d == dst_d:
                    continue
                dst_buffer = _create_buffer(
                    dst_d,
                    output_id,
                    is_input=False,
                    stage_id=len(workload_per_stage),
                    creation_device=d,
                )
                # make sure adding the pair of send and recvs at the same time
                per_device_comm_ops[d].append(
                    CommOp(CommType.SEND, dst_d, src_buffer)
                )
                per_device_comm_ops[dst_d].append(
                    CommOp(CommType.RECV, d, dst_buffer)
                )
        # inject communication instructions
        comm_key = f"O"
        for d in all_devices:
            if d in per_device_comm_ops:
                per_device_instructions[d].append(
                    CommLaunchInstr(comm_key, per_device_comm_ops[d])
                )
                pending_comm_keys[d].add(comm_key)

    def _wait_for_output_comm():
        for d in all_devices:
            for comm_key in pending_comm_keys[d]:
                per_device_instructions[d].append(CommWaitInstr(comm_key))
            pending_comm_keys[d].clear()

    def _gen_fw_reduction_instructions():
        # add reduction instructions
        per_device_red_ops = defaultdict(list)
        # construct device to output map
        device_to_output_map = defaultdict(list)
        for output_id, d in output_to_device_map.items():
            device_to_output_map[d].append(output_id)
        for final_d, output_ids in device_to_output_map.items():
            # (seq_id, head_id, block_id) -> version -> [out_buffer, lse_buffer]
            buffer_pairs: Dict[Tuple, Dict] = {}
            seq_block_id_to_output_lse_id = {}
            for output_id in output_ids:
                out_meta = workload.block_mapping.output_id_to_meta[output_id]
                seq_id = out_meta.seq_id
                head_id = out_meta.head_id
                block_id = out_meta.block_id
                block_type = out_meta.type
                (
                    out_buffers,
                    out_creation_stages,
                    out_creation_devices,
                    replica_ids,
                ) = _get_buffer_all_versions(
                    final_d, output_id, is_input=False
                )
                versions = zip(
                    out_creation_stages, out_creation_devices, replica_ids
                )
                if (
                    seq_id,
                    head_id,
                    block_id,
                ) not in seq_block_id_to_output_lse_id:
                    seq_block_id_to_output_lse_id[
                        (seq_id, head_id, block_id)
                    ] = [
                        None,
                        None,
                    ]
                if block_type == BlockType.Out:
                    seq_block_id_to_output_lse_id[(seq_id, head_id, block_id)][
                        0
                    ] = output_id
                elif block_type == BlockType.LSE:
                    seq_block_id_to_output_lse_id[(seq_id, head_id, block_id)][
                        1
                    ] = output_id
                for buffer, version in zip(out_buffers, versions):
                    if (seq_id, head_id, block_id) not in buffer_pairs:
                        buffer_pairs[(seq_id, head_id, block_id)] = {}
                    if (
                        version
                        not in buffer_pairs[(seq_id, head_id, block_id)]
                    ):
                        buffer_pairs[(seq_id, head_id, block_id)][version] = [
                            None,
                            None,
                        ]
                    if block_type == BlockType.Out:
                        buffer_pairs[(seq_id, head_id, block_id)][version][
                            0
                        ] = buffer
                    elif block_type == BlockType.LSE:
                        buffer_pairs[(seq_id, head_id, block_id)][version][
                            1
                        ] = buffer
            if logging_enabled:
                _logger_debug(
                    f"Device {final_d} has {len(buffer_pairs)} "
                    "pairs of output buffers to reduce."
                )
            for (
                seq_id,
                head_id,
                block_id,
            ), version_buffer_pairs in buffer_pairs.items():
                out_id, lse_id = seq_block_id_to_output_lse_id[
                    (seq_id, head_id, block_id)
                ]
                src_buffers = []
                final_out_buffer_idx = (
                    workload.block_mapping.output_id_to_buffer_index[final_d][
                        out_id
                    ]
                )
                final_lse_buffer_idx = (
                    workload.block_mapping.output_id_to_buffer_index[final_d][
                        lse_id
                    ]
                )
                n_tokens = workload.block_mapping.output_id_to_meta[
                    out_id
                ].n_tokens
                assert (
                    n_tokens
                    == workload.block_mapping.output_id_to_meta[
                        lse_id
                    ].n_tokens
                )
                for out_buffer, lse_buffer in version_buffer_pairs.values():
                    src_buffers.append((out_buffer, lse_buffer))
                if src_buffers:
                    per_device_red_ops[final_d].append(
                        ReductionOp(
                            src_buffers,
                            (
                                BufferBlock(
                                    BufferType.LOCAL_OUT,
                                    final_out_buffer_idx,
                                    n_tokens,
                                ),
                                BufferBlock(
                                    BufferType.LOCAL_LSE,
                                    final_lse_buffer_idx,
                                    n_tokens,
                                ),
                            ),
                        )
                    )
        # inject reduction instructions
        for d, red_ops in per_device_red_ops.items():
            per_device_instructions[d].append(
                AttnReductionInstr(red_ops, output_is_unpadded=1)
            )

    def _gen_bw_reduction_instructions():
        # add reduction instructions
        per_device_red_ops = defaultdict(list)
        # construct device to output map
        device_to_output_map = defaultdict(list)
        for output_id, d in output_to_device_map.items():
            device_to_output_map[d].append(output_id)
        for final_d, output_ids in device_to_output_map.items():
            # (seq_id, block_id) -> version -> [dq_buffer, dkv_buffer]
            local_buffers = {
                BufferType.LOCAL_dQ: defaultdict(list),
                BufferType.LOCAL_dKV: defaultdict(list),
            }
            for output_id in output_ids:
                out_meta = workload.block_mapping.output_id_to_meta[output_id]
                block_type = out_meta.type
                (
                    out_buffers,
                    _,
                    _,
                    _,
                ) = _get_buffer_all_versions(
                    final_d, output_id, is_input=False
                )
                if block_type == BlockType.dQ:
                    local_buffers[BufferType.LOCAL_dQ][output_id] = out_buffers
                elif block_type == BlockType.dKV:
                    local_buffers[BufferType.LOCAL_dKV][
                        output_id
                    ] = out_buffers
            for buffer_type in local_buffers.keys():
                for output_id, src_buffers in local_buffers[
                    buffer_type
                ].items():
                    n_tokens = workload.block_mapping.output_id_to_meta[
                        output_id
                    ].n_tokens
                    dst_buffer = BufferBlock(
                        buffer_type,
                        workload.block_mapping.output_id_to_buffer_index[
                            final_d
                        ][output_id],
                        n_tokens,
                    )
                    per_device_red_ops[final_d].append(
                        SumOp(src_buffers, dst_buffer)
                    )
        # inject reduction instructions
        for d, red_ops in per_device_red_ops.items():
            per_device_instructions[d].append(
                SumInstr(red_ops, output_is_unpadded=1)
            )

    def _create_execution_plan(is_forward: bool):
        device_to_input_map = defaultdict(list)
        for input_id, d in input_to_device_map.items():
            device_to_input_map[d].append(input_id)
        device_to_output_map = defaultdict(list)
        for output_id, d in output_to_device_map.items():
            device_to_output_map[d].append(output_id)
        exec_plans = {}

        for d in all_devices:
            buffer_info = {}
            # for each buffer type, get the number of blocks, size of each block,
            # and dtype
            block_input_types = (
                _FW_BLOCK_INPUT_TYPES if is_forward else _BW_BLOCK_INPUT_TYPES
            )
            block_output_types = (
                _FW_BLOCK_OUTPUT_TYPES
                if is_forward
                else _BW_BLOCK_OUTPUT_TYPES
            )
            block_types = block_input_types.union(block_output_types)
            for block_type in block_types:
                if block_type in (
                    _FW_BLOCK_INPUT_TYPES
                    if is_forward
                    else _BW_BLOCK_INPUT_TYPES
                ):
                    id_to_meta = workload.block_mapping.input_id_to_meta
                    id_to_device_map = input_to_device_map
                else:
                    id_to_meta = workload.block_mapping.output_id_to_meta
                    id_to_device_map = output_to_device_map

                all_local_type_data_ids = [
                    i
                    for i in id_to_device_map.keys()
                    if id_to_meta[i].type == block_type
                    and id_to_device_map[i] == d
                ]
                local_total_n_tokens = sum(
                    id_to_meta[i].n_tokens for i in all_local_type_data_ids
                )
                if logging_enabled:
                    _logger_debug(
                        f"Device {d}, block type {block_type}, "
                        f"has {len(all_local_type_data_ids)} blocks."
                    )
                n_blocks = len(all_local_type_data_ids)
                if n_blocks == 0:
                    # no need to create buffer for this block type
                    continue
                meta = id_to_meta[all_local_type_data_ids[0]]
                block_size = meta.block_size
                numel = meta.numel()
                buffer_type = _LOCAL_BLOCK_TYPE_TO_BUFFER_TYPE[block_type]
                if buffer_type in _UNPADDED_BUFFER_TYPES:
                    if block_type in _QKV_ORDERED_BUFFER_BLOCK_TYPES:
                        buffer_shape = (
                            local_total_n_tokens,
                            *meta.per_token_shape,
                        )
                        per_token_shape = meta.per_token_shape
                    elif block_type in _LSE_ORDERED_BUFFER_BLOCK_TYPES:
                        buffer_shape = (
                            meta.per_token_shape[0],
                            local_total_n_tokens,
                        )
                        per_token_shape = (1,)
                    else:
                        raise ValueError(f"Unknown buffer type: {block_type}")
                else:
                    if block_type in _QKV_ORDERED_BUFFER_BLOCK_TYPES:
                        buffer_shape = (
                            n_blocks,
                            block_size,
                            *meta.per_token_shape,
                        )
                        per_token_shape = meta.per_token_shape
                    elif block_type in _LSE_ORDERED_BUFFER_BLOCK_TYPES:
                        buffer_shape = (
                            meta.per_token_shape[0],
                            n_blocks,
                            block_size,
                        )
                        per_token_shape = (1,)
                    else:
                        raise ValueError(f"Unknown buffer type: {block_type}")
                if logging_enabled:
                    _logger_debug(
                        f"Creating buffer info: Device {d}, buffer type {buffer_type}, "
                        f"buffer shape: {buffer_shape}"
                    )
                buffer_info[buffer_type] = BufferInfo(
                    n_blocks,
                    block_size,
                    numel,
                    workload.block_mapping.buffer_type_to_dtype[buffer_type],
                    buffer_shape=buffer_shape,
                    per_token_shape=per_token_shape,
                )
            # add intermediate buffers
            buffer_metas = {k: None for k in _BUFFER_TYPE_TO_BLOCK_TYPE.keys()}
            for (
                input_id,
                meta,
            ) in workload.block_mapping.input_id_to_meta.items():
                if meta.type in _BUFFER_BLOCK_TYPE_TO_BUFFER_TYPE:
                    buffer_type = _BUFFER_BLOCK_TYPE_TO_BUFFER_TYPE[meta.type]
                    buffer_metas[buffer_type] = meta
                    if all(
                        buffer_metas[k] is not None
                        for k in _BUFFER_TYPE_TO_BLOCK_TYPE.keys()
                    ):
                        break
            for (
                output_id,
                meta,
            ) in workload.block_mapping.output_id_to_meta.items():
                if meta.type in _BUFFER_BLOCK_TYPE_TO_BUFFER_TYPE:
                    buffer_type = _BUFFER_BLOCK_TYPE_TO_BUFFER_TYPE[meta.type]
                    buffer_metas[buffer_type] = meta
                    if all(
                        buffer_metas[k] is not None
                        for k in _BUFFER_TYPE_TO_BLOCK_TYPE.keys()
                    ):
                        break
            for buffer_type, meta in buffer_metas.items():
                if meta is None:
                    continue
                block_manager = _BUFFER_TYPE_TO_MANAGERS[buffer_type][d]
                block_size = meta.block_size
                numel = meta.numel()
                n_blocks = len(block_manager.blocks)
                if meta.type in _QKV_ORDERED_BUFFER_BLOCK_TYPES:
                    buffer_shape = (
                        n_blocks,
                        block_size,
                        *meta.per_token_shape,
                    )
                    per_token_shape = meta.per_token_shape
                elif meta.type in _LSE_ORDERED_BUFFER_BLOCK_TYPES:
                    buffer_shape = (
                        meta.per_token_shape[0],
                        n_blocks,
                        block_size,
                    )
                    per_token_shape = (1,)
                if logging_enabled:
                    _logger_debug(
                        f"Creating buffer info: Device {d}, buffer type {buffer_type}, "
                        f"buffer shape: {buffer_shape}"
                    )
                buffer_info[buffer_type] = BufferInfo(
                    n_blocks,
                    block_size,
                    numel,
                    workload.block_mapping.buffer_type_to_dtype[buffer_type],
                    buffer_shape=buffer_shape,
                    per_token_shape=per_token_shape,
                )

            exec_plan = ExecutionPlan(
                per_device_instructions[d],
                context.n_nodes,
                context.n_devices_per_node,
                d[0],
                d[1],
                len(workload_per_stage),
                None,  # to be filled in compiler
                buffer_info,
            )
            exec_plans[d] = exec_plan
        return exec_plans

    # _inject_barrier()

    _copy_input_to_local_buffer()

    start_time = time.time()
    for stage_id, stage in enumerate(workload_per_stage):
        per_stage_instructions = defaultdict(list)  # d -> List[Instr]

        # wait for last stage communication
        _wait_for_input_comm(stage_id)

        # first, get all required data communication across devices
        # for the NEXT stage
        _gen_input_comm_instrs_for_stage(per_stage_instructions, stage_id)

        if logging_enabled:
            _logger_debug(f"Stage {stage_id}, initial KV blocks:")
            for d, manager in kv_block_managers.items():
                _logger_debug(f"Device {d}: {manager.blocks}")
            _logger_debug(f"Stage {stage_id}, initial Q blocks:")
            for d, manager in q_block_managers.items():
                _logger_debug(f"Device {d}: {manager.blocks}")

        if len(workload_per_stage) == 1:
            # only one stage, directly wait for input comm
            _wait_for_input_comm(stage_id)

        # add attn instructions
        if is_forward:
            _gen_attn_fw_instr_for_stage(
                per_stage_instructions, stage_id, stage
            )
        else:
            _gen_attn_bw_instr_for_stage(
                per_stage_instructions, stage_id, stage
            )

        # if logging_enabled:
        #     _logger_debug(f"Stage {stage_id}, KV blocks after scheduling:")
        #     for d, manager in kv_block_managers.items():
        #         _logger_debug(f"Device {d}: {manager.blocks}")
        # free input blocks that are no longer needed
        if is_forward:
            managers_list = [q_block_managers, kv_block_managers]
        else:
            managers_list = [
                q_block_managers,
                kv_block_managers,
                out_block_managers,
                lse_block_managers,
                dout_block_managers,
            ]
        for man_id, managers in enumerate(managers_list):
            # if logging_enabled:
            #     if man_id == 0:
            #         _logger_debug(f"Stage {stage_id}, before freeing Q blocks:")
            #         for d, manager in managers.items():
            #             _logger_debug(f"Device {d}: {manager.blocks}")
            #     else:
            #         _logger_debug(
            #             f"Stage {stage_id}, before freeing KV blocks:"
            #         )
            #         for d, manager in managers.items():
            #             _logger_debug(f"Device {d}: {manager.blocks}")
            for d, manager in managers.items():
                for block_key in manager.blocks:
                    if block_key is None:
                        continue
                    data_id, is_input, creation_device, stage, _ = block_key
                    if data_id is not None and is_input:
                        if (
                            data_id not in per_device_input_live_range[d]
                            or per_device_input_live_range[d][data_id][1]
                            <= stage_id
                        ):
                            manager.free_block(
                                data_id,
                                is_input=is_input,
                                creation_device=creation_device,
                                stage_id=stage,
                            )

        # add instructions to the per_device_instructions
        for d in all_devices:
            per_device_instructions[d].extend(per_stage_instructions[d])

    end_time = time.time()
    if logging_enabled:
        _logger_debug(
            f"Scheduling for {len(workload_per_stage)} stages took "
            f"{end_time - start_time} seconds"
        )

    _gen_output_comm_instrs()

    _wait_for_output_comm()

    if is_forward:
        _gen_fw_reduction_instructions()
    else:
        _gen_bw_reduction_instructions()

    exec_plans = _create_execution_plan(is_forward=is_forward)

    if SCHEDULE_LOG_BUFFER_INFO:
        for d, exec_plan in exec_plans.items():
            exec_plan: ExecutionPlan
            _logger_debug(f"Device {d}, buffer info:")
            for buffer_type, buffer_info in exec_plan.buffer_info.items():
                _logger_debug(
                    f"\tBuffer type: {buffer_type}, "
                    f"n_blocks: {buffer_info.n_blocks}, "
                    f"block_size: {buffer_info.block_size}, "
                    f"numel: {buffer_info.block_numel}, "
                    f"buffer_shape: {buffer_info.buffer_shape}"
                )

    # aux output for debugging
    work_to_stage_map = {}
    for stage_id, stage in enumerate(workload_per_stage):
        for d, workload_ids in stage.items():
            for workload_id in workload_ids:
                work_to_stage_map[workload_id] = stage_id
    return exec_plans, work_to_stage_map
