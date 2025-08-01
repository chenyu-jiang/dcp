from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import pytest
import torch

from dcp.core.common import ExecutionContext
from dcp.core.block_table import BlockType, WorkloadSpec
from dcp.core.compiler import CompilerConfig, InstrCompiler
from dcp.core.cost_model import (
    AttnRooflineCostModel,
    CommunicationCostModel,
)
from dcp.core.instructions import *  # noqa: F403
from dcp.data.dataloader import generate_workload
from dcp.utils.logger import create_logger

logger = create_logger(__name__, log_file="test_compiler.log")


def is_matching_comm_op(
    device_a: Tuple[int, int],
    comm_op_a: CommOp,
    device_b: Tuple[int, int],
    comm_op_b: CommOp,
    buffer_info: Dict[str, BufferInfo],
):
    if comm_op_a.comm_type == comm_op_b.comm_type:
        return False
    if not (comm_op_b.peer == device_a) or not (comm_op_a.peer == device_b):
        return False
    # check buffer info
    buffer_block_numel_a = buffer_info[
        comm_op_a.buffer_block.buffer_type
    ].block_numel
    buffer_dtype_a = buffer_info[comm_op_a.buffer_block.buffer_type].dtype
    buffer_block_numel_b = buffer_info[
        comm_op_b.buffer_block.buffer_type
    ].block_numel
    buffer_dtype_b = buffer_info[comm_op_b.buffer_block.buffer_type].dtype
    if (
        buffer_block_numel_a != buffer_block_numel_b
        or buffer_dtype_a != buffer_dtype_b
    ):
        return False
    return True


def find_matching_communications(
    instr_per_device: Dict[Tuple[int, int], CommLaunchInstr],
    buffer_info: Dict[str, BufferInfo],
):
    logger.info("Find matching communications...")
    for d, instr in instr_per_device.items():
        logger.info("Device {}: {}".format(d, instr))

    comm_op_keys = set()
    for d, instr in instr_per_device.items():
        comm_op_keys.add(instr.key)

    for key in comm_op_keys:
        logger.debug("Trying to match key: {}".format(key))
        instrs_per_device_with_key = {
            d: instr
            for d, instr in instr_per_device.items()
            if instr.key == key
        }
        comm_op_ptrs = {d: 0 for d in instrs_per_device_with_key.keys()}
        matchings = defaultdict(
            dict
        )  # device -> curr_comm_op_idx -> peer_comm_op_idx

        def _all_comm_ops_done():
            return all(
                comm_op_ptrs[d] == len(instr.comm_ops)
                for d, instr in instrs_per_device_with_key.items()
            )

        matched = True
        while not _all_comm_ops_done():
            progress_made = False
            for d, instr in instrs_per_device_with_key.items():
                comm_ops = instr.comm_ops
                curr_comm_op_ptr = comm_op_ptrs[d]
                if curr_comm_op_ptr == len(comm_ops):
                    continue
                comm_op = comm_ops[curr_comm_op_ptr]
                peer = comm_op.peer
                if peer not in instrs_per_device_with_key:
                    matched = False
                    break
                peer_instr = instrs_per_device_with_key[peer]
                peer_comm_ops = peer_instr.comm_ops
                peer_comm_op_ptr = comm_op_ptrs[peer]
                if peer_comm_op_ptr == len(peer_comm_ops):
                    matched = False
                    break
                peer_comm_op = peer_comm_ops[peer_comm_op_ptr]
                if is_matching_comm_op(
                    d, comm_op, peer, peer_comm_op, buffer_info
                ):
                    matchings[d][curr_comm_op_ptr] = peer_comm_op_ptr
                    matchings[peer][peer_comm_op_ptr] = curr_comm_op_ptr
                    logger.info(
                        "Matched: Device {}: {} <-> Device {}: {}".format(
                            d, comm_op, peer, peer_comm_op
                        )
                    )
                    comm_op_ptrs[d] += 1
                    comm_op_ptrs[peer] += 1
                    progress_made = True
                    break
            if not progress_made:
                matched = False
            if not matched:
                break
        if matched:
            logger.debug("Matched key: {}".format(key))
            return True, matchings, key
    return False, None, None


BLOCK_TYPE_TO_BUFFER_TYPE = {
    BlockType.Q: BufferType.LOCAL_Q,
    BlockType.KV: BufferType.LOCAL_KV,
    BlockType.Out: BufferType.LOCAL_OUT,
    BlockType.LSE: BufferType.LOCAL_LSE,
    BlockType.dQ: BufferType.LOCAL_dQ,
    BlockType.dKV: BufferType.LOCAL_dKV,
    BlockType.dOut: BufferType.LOCAL_dOUT,
}


class DummyExecutor:
    def __init__(
        self,
        workload_spec: WorkloadSpec,
        execution_plans: Dict[Tuple[int, int], ExecutionPlan],
    ):
        self.workload_spec = workload_spec
        self.execution_plans = execution_plans
        self.buffers = defaultdict(
            lambda: defaultdict(list)
        )  # d -> str -> List[]
        self.q_idx_to_out_indices = defaultdict(lambda: (None, None))
        self.q_idx_to_dq_indices = defaultdict(lambda: None)
        self.q_idx_to_dout_indices = defaultdict(lambda: None)
        self.kv_idx_to_dkv_indices = defaultdict(lambda: None)
        self._init_buffers()

    def _init_buffers(self):
        logger.info("Initialize buffers..." + "-" * 20)
        # initialize input and output buffers
        for (
            d,
            input_id_mapping,
        ) in self.workload_spec.block_mapping.input_id_to_buffer_index.items():
            for input_id, buffer_index in input_id_mapping.items():
                buffer_type = BLOCK_TYPE_TO_BUFFER_TYPE[
                    self.workload_spec.block_mapping.input_id_to_meta[
                        input_id
                    ].type
                ]
                while len(self.buffers[d][buffer_type]) < buffer_index + 1:
                    self.buffers[d][buffer_type].append(None)
                self.buffers[d][buffer_type][buffer_index] = input_id
                logger.info(
                    "Device {}: {}[{}] <- I{}".format(
                        d, buffer_type, buffer_index, input_id
                    )
                )
        for (
            d,
            output_id_mapping,
        ) in (
            self.workload_spec.block_mapping.output_id_to_buffer_index.items()
        ):
            for output_id, buffer_index in output_id_mapping.items():
                buffer_type = BLOCK_TYPE_TO_BUFFER_TYPE[
                    self.workload_spec.block_mapping.output_id_to_meta[
                        output_id
                    ].type
                ]
                while len(self.buffers[d][buffer_type]) < buffer_index + 1:
                    self.buffers[d][buffer_type].append(None)
        # initialize buffers used by exec_plan
        for d, exec_plan in self.execution_plans.items():
            for buffer_name, buffer_info in exec_plan.buffer_info.items():
                if buffer_name in BLOCK_TYPE_TO_BUFFER_TYPE.values():
                    continue
                self.buffers[d][buffer_name] = [None] * buffer_info.n_blocks
                logger.info(
                    "Device {}: {}[0:{}] <- None".format(
                        d, buffer_name, buffer_info.n_blocks
                    )
                )
        for (
            workload_meta
        ) in self.workload_spec.block_mapping.work_id_to_meta.values():
            q_id = workload_meta.q_id
            kv_id = workload_meta.kv_id
            out_id = workload_meta.out_id
            lse_id = workload_meta.lse_id
            dq_id = workload_meta.dq_id
            dkv_id = workload_meta.dkv_id
            dout_id = workload_meta.dout_id
            self.q_idx_to_out_indices[q_id] = (out_id, lse_id)
            if dq_id is not None:
                self.q_idx_to_dq_indices[q_id] = dq_id
            if dkv_id is not None:
                self.kv_idx_to_dkv_indices[kv_id] = dkv_id
            if dout_id is not None:
                self.q_idx_to_dout_indices[q_id] = dout_id
        logger.info("Finished initializing buffers..." + "-" * 20)

    def execute(self):
        instr_ptr_per_device = {d: 0 for d in self.execution_plans}

        def _execution_done():
            return all(
                instr_ptr_per_device[d]
                == len(self.execution_plans[d].instructions)
                for d in self.execution_plans.keys()
            )

        def _get_meta(block_id, is_input):
            if is_input:
                return self.workload_spec.block_mapping.input_id_to_meta[
                    block_id
                ]
            else:
                return self.workload_spec.block_mapping.output_id_to_meta[
                    block_id
                ]

        def _check_list_of_blocks(
            blocks: List[BufferBlock],
            is_input,
            require_nonempty=True,
            check_total_length=None,
        ):
            block_total_seqlen = 0
            for block in blocks:
                assert block.index < len(
                    self.buffers[d][block.buffer_type]
                ), f"Device {d}: {block.buffer_type}[{block.index}] is None"
                data_id = self.buffers[d][block.buffer_type][block.index]
                if require_nonempty:
                    assert (
                        data_id is not None
                    ), f"Device {d}: {block.buffer_type}[{block.index}] is None"
                    meta = _get_meta(data_id, is_input)
                    block_total_seqlen += meta.n_tokens
            if check_total_length is not None and require_nonempty:
                assert (
                    block_total_seqlen == check_total_length
                ), f"Device {d}: Total seqlen mismatch: {block_total_seqlen} != {check_total_length}"

        def _check_block_table(
            seqlens, block_table, is_input, require_nonempty=True
        ):
            for b, seqlen in enumerate(seqlens):
                cache_blocks = block_table[b]
                _check_list_of_blocks(
                    cache_blocks,
                    is_input,
                    require_nonempty=require_nonempty,
                    check_total_length=seqlen,
                )

        def _check_attn_mask(attn_mask, seqlens_q, seqlens_kv):
            cu_seqlens_q = np.cumsum([0] + seqlens_q)
            if isinstance(attn_mask, list):
                attn_mask_array = np.stack(
                    [x.numpy() for x in attn_mask], axis=0
                )
            else:
                attn_mask_array = attn_mask.numpy()
            if attn_mask_array.ndim == 3:
                # shape: 2, 2, sum(seqlens_q)
                assert attn_mask_array.shape[0] == 2
                assert attn_mask_array.shape[1] == 2
                assert attn_mask_array.shape[2] == sum(seqlens_q)
                for b, seqlen_q in enumerate(seqlens_q):
                    for i in range(seqlen_q):
                        assert (
                            attn_mask_array[:, :, cu_seqlens_q[b] + i]
                            <= seqlens_kv[b]
                        ).all()
                        assert (
                            attn_mask_array[:, 0] <= attn_mask_array[:, 1]
                        ).all()
            elif attn_mask_array.ndim == 2:
                # shape: 2, sum(seqlens_q)
                assert attn_mask_array.shape[0] == 2
                assert attn_mask_array.shape[1] == sum(seqlens_q)
                for b, seqlen_q in enumerate(seqlens_q):
                    for i in range(seqlen_q):
                        assert (
                            attn_mask_array[:, cu_seqlens_q[b] + i]
                            <= seqlens_kv[b]
                        ).all()
                        assert (attn_mask_array[0] <= attn_mask_array[1]).all()
            else:
                raise ValueError(
                    f"Unknown attn_mask shape: {attn_mask_array.shape}"
                )

        while not _execution_done():
            # first check if any device has comp instruction
            is_comm_op = {d: False for d in self.execution_plans.keys()}
            for d, exec_plan in self.execution_plans.items():
                instr_ptr = instr_ptr_per_device[d]
                if instr_ptr == len(exec_plan.instructions):
                    continue
                instr = exec_plan.instructions[instr_ptr]
                logger.info("## Encountered ## Device {}: {}".format(d, instr))
                if isinstance(instr, MemcpyInstr):
                    # simulate memcpy
                    for src_block, dst_block in instr.src_dst_pairs:
                        # logger.info("Memcpy: {} -> {}".format(src_block, dst_block))
                        src_buffer = self.buffers[d][src_block.buffer_type][
                            src_block.index
                        ]
                        # check they can be copied
                        assert (
                            exec_plan.buffer_info[
                                dst_block.buffer_type
                            ].block_size
                            == exec_plan.buffer_info[
                                src_block.buffer_type
                            ].block_size
                        )
                        assert (
                            exec_plan.buffer_info[
                                dst_block.buffer_type
                            ].block_numel
                            == exec_plan.buffer_info[
                                src_block.buffer_type
                            ].block_numel
                        )
                        assert (
                            exec_plan.buffer_info[dst_block.buffer_type].dtype
                            == exec_plan.buffer_info[
                                src_block.buffer_type
                            ].dtype
                        ), (
                            f"Device {d}: {dst_block.buffer_type} dtype: {exec_plan.buffer_info[dst_block.buffer_type].dtype}, "
                            f"{src_block.buffer_type} dtype: {exec_plan.buffer_info[src_block.buffer_type].dtype}"
                        )
                        assert src_buffer is not None
                        self.buffers[d][dst_block.buffer_type][
                            dst_block.index
                        ] = src_buffer
                    logger.info(
                        "## Executed ## Device {}: {}".format(d, instr)
                    )
                    # increment instr_ptr
                    instr_ptr_per_device[d] += 1
                elif isinstance(instr, AttnInstr):
                    # first check if block table is valid
                    assert (
                        len(instr.seqlens_q)
                        == len(instr.q_block_table)
                        == len(instr.seqlens_kv)
                        == len(instr.kv_block_table)
                        == len(instr.out_block_table)
                        == len(instr.lse_block_table)
                    ), (
                        "Length mismatch: seqlens_q: {}, q_block_table: {}, "
                        "seqlens_kv: {}, kv_block_table: {}, "
                        "out_block_table: {}, lse_block_table: {}".format(
                            len(instr.seqlens_q),
                            len(instr.q_block_table),
                            len(instr.seqlens_kv),
                            len(instr.kv_block_table),
                            len(instr.out_block_table),
                            len(instr.lse_block_table),
                        )
                    )

                    # check if seqlens_q is valid
                    logger.info(
                        "Buffer Q: {}".format(
                            self.buffers[d][BufferType.BUFFER_Q]
                        )
                    )
                    logger.info(
                        "Buffer KV: {}".format(
                            self.buffers[d][BufferType.BUFFER_KV]
                        )
                    )
                    logger.info(
                        "Buffer LSE: {}".format(
                            self.buffers[d][BufferType.BUFFER_LSE]
                        )
                    )
                    _check_block_table(
                        instr.seqlens_q,
                        instr.q_block_table,
                        is_input=True,
                    )
                    _check_block_table(
                        instr.seqlens_kv,
                        instr.kv_block_table,
                        is_input=True,
                    )
                    _check_block_table(
                        instr.seqlens_q,
                        instr.out_block_table,
                        is_input=False,
                        require_nonempty=False,
                    )
                    _check_attn_mask(
                        instr.attn_mask, instr.seqlens_q, instr.seqlens_kv
                    )
                    # fill in the output buffers
                    for b in range(len(instr.out_block_table)):
                        q_blocks = instr.q_block_table[b]
                        for ith_block_in_table, block in enumerate(q_blocks):
                            assert block.buffer_type == BufferType.BUFFER_Q
                            q_id = self.buffers[d][block.buffer_type][
                                block.index
                            ]
                            assert q_id is not None
                            output_id, lse_id = self.q_idx_to_out_indices[q_id]
                            assert output_id is not None
                            assert lse_id is not None
                            out_block = instr.out_block_table[b][
                                ith_block_in_table
                            ]
                            assert (
                                out_block.buffer_type == BufferType.BUFFER_OUT
                            )
                            lse_block = instr.lse_block_table[b][
                                ith_block_in_table
                            ]
                            assert (
                                lse_block.buffer_type == BufferType.BUFFER_LSE
                            )
                            self.buffers[d][out_block.buffer_type][
                                out_block.index
                            ] = output_id
                            self.buffers[d][lse_block.buffer_type][
                                lse_block.index
                            ] = lse_id
                        # checks for kv blocks
                        kv_blocks = instr.kv_block_table[b]
                        for block in kv_blocks:
                            assert block.buffer_type == BufferType.BUFFER_KV
                    logger.info(
                        "## Executed ## Device {}: {}".format(d, instr)
                    )
                    # increment instr_ptr
                    instr_ptr_per_device[d] += 1
                elif isinstance(instr, AttnBackwardInstr):
                    # first check if block table is valid
                    assert (
                        len(instr.seqlens_q)
                        == len(instr.q_block_table)
                        == len(instr.seqlens_kv)
                        == len(instr.kv_block_table)
                        == len(instr.out_block_table)
                        == len(instr.dq_block_table)
                        == len(instr.dkv_block_table)
                    ), (
                        "Length mismatch: seqlens_q: {}, q_block_table: {}, "
                        "seqlens_kv: {}, kv_block_table: {}, "
                        "out_block_table: {}, dq_block_table: {}, "
                        "dkv_block_table: {}".format(
                            len(instr.seqlens_q),
                            len(instr.q_block_table),
                            len(instr.seqlens_kv),
                            len(instr.kv_block_table),
                            len(instr.out_block_table),
                            len(instr.dq_block_table),
                            len(instr.dkv_block_table),
                        )
                    )
                    _check_block_table(
                        instr.seqlens_q,
                        instr.q_block_table,
                        is_input=True,
                    )
                    _check_block_table(
                        instr.seqlens_kv,
                        instr.kv_block_table,
                        is_input=True,
                    )
                    _check_block_table(
                        instr.seqlens_q,
                        instr.out_block_table,
                        is_input=True,
                    )
                    _check_block_table(
                        instr.seqlens_q,
                        instr.dq_block_table,
                        is_input=False,
                        require_nonempty=False,
                    )
                    _check_block_table(
                        instr.seqlens_kv,
                        instr.dkv_block_table,
                        is_input=False,
                        require_nonempty=False,
                    )
                    # fill in the output buffers
                    for b in range(len(instr.dq_block_table)):
                        # dq buffer
                        q_blocks = instr.q_block_table[b]
                        kv_blocks = instr.kv_block_table[b]
                        out_blocks = instr.out_block_table[b]
                        dq_blocks = instr.dq_block_table[b]
                        dkv_blocks = instr.dkv_block_table[b]
                        assert len(q_blocks) == len(dq_blocks)
                        assert len(kv_blocks) == len(dkv_blocks)
                        assert len(out_blocks) == len(dq_blocks)
                        for ith_block_in_table, block in enumerate(q_blocks):
                            assert block.buffer_type == BufferType.BUFFER_Q, (
                                f"Device {d}: {block.buffer_type} is not "
                                f"BUFFER_Q"
                            )
                            q_id = self.buffers[d][block.buffer_type][
                                block.index
                            ]
                            assert q_id is not None
                            output_id, lse_id = self.q_idx_to_out_indices[q_id]
                            assert output_id is not None
                            assert lse_id is not None
                            out_block = instr.out_block_table[b][
                                ith_block_in_table
                            ]
                            assert (
                                out_block.buffer_type == BufferType.BUFFER_OUT
                            )
                            out_id_from_buffer = self.buffers[d][
                                out_block.buffer_type
                            ][out_block.index]
                            assert out_id_from_buffer == output_id
                            lse_id_from_buffer = self.buffers[d][
                                BufferType.BUFFER_LSE
                            ][out_block.index]
                            assert lse_id_from_buffer == lse_id
                            dq_block = instr.dq_block_table[b][
                                ith_block_in_table
                            ]
                            assert dq_block.buffer_type == BufferType.BUFFER_dQ
                            dq_id = self.q_idx_to_dq_indices[q_id]
                            assert dq_id is not None
                            self.buffers[d][dq_block.buffer_type][
                                dq_block.index
                            ] = dq_id
                        # kv blocks
                        for ith_block_in_table, block in enumerate(kv_blocks):
                            assert block.buffer_type == BufferType.BUFFER_KV
                            kv_id = self.buffers[d][block.buffer_type][
                                block.index
                            ]
                            assert kv_id is not None
                            dkv_id = self.kv_idx_to_dkv_indices[kv_id]
                            assert dkv_id is not None
                            dkv_block = instr.dkv_block_table[b][
                                ith_block_in_table
                            ]
                            assert (
                                dkv_block.buffer_type == BufferType.BUFFER_dKV
                            )
                            self.buffers[d][dkv_block.buffer_type][
                                dkv_block.index
                            ] = dkv_id
                    logger.info(
                        "## Executed ## Device {}: {}".format(d, instr)
                    )
                    # increment instr_ptr
                    instr_ptr_per_device[d] += 1
                elif isinstance(instr, AttnReductionInstr):
                    # simulate reduction
                    src_out_buffer_type = None
                    src_lse_buffer_type = None
                    for red_op in instr.ops:
                        src_buffers, (dst_out_buffer, dst_lse_buffer) = (
                            red_op.src_buffers,
                            red_op.dst_buffer,
                        )
                        for src_out_buffer, src_lse_buffer in src_buffers:
                            if src_out_buffer_type is None:
                                src_out_buffer_type = (
                                    src_out_buffer.buffer_type
                                )
                            else:
                                assert (
                                    src_out_buffer_type
                                    == src_out_buffer.buffer_type
                                )
                            if src_lse_buffer_type is None:
                                src_lse_buffer_type = (
                                    src_lse_buffer.buffer_type
                                )
                            else:
                                assert (
                                    src_lse_buffer_type
                                    == src_lse_buffer.buffer_type
                                )
                        # check dst_buffer is not aleady populated
                        assert (
                            len(self.buffers[d][dst_out_buffer.buffer_type])
                            > dst_out_buffer.index
                        )
                        assert (
                            len(self.buffers[d][dst_lse_buffer.buffer_type])
                            > dst_lse_buffer.index
                        )
                        out_id = self.buffers[d][dst_out_buffer.buffer_type][
                            dst_out_buffer.index
                        ]
                        lse_id = self.buffers[d][dst_lse_buffer.buffer_type][
                            dst_lse_buffer.index
                        ]
                        # check they all have the same out id
                        out_id = None
                        lse_id = None
                        for src_out_buffer, src_lse_buffer in src_buffers:
                            curr_out_id = self.buffers[d][
                                src_out_buffer.buffer_type
                            ][src_out_buffer.index]
                            curr_lse_id = self.buffers[d][
                                src_lse_buffer.buffer_type
                            ][src_lse_buffer.index]
                            assert curr_out_id is not None
                            assert curr_lse_id is not None
                            if out_id is not None:
                                assert curr_out_id == out_id
                            else:
                                out_id = curr_out_id
                            if lse_id is not None:
                                assert curr_lse_id == lse_id
                            else:
                                lse_id = curr_lse_id
                    logger.info(
                        "## Executed ## Device {}: {}".format(d, instr)
                    )
                    # increment instr_ptr
                    instr_ptr_per_device[d] += 1
                elif isinstance(instr, SumInstr):
                    for red_op in instr.ops:
                        src_buffers, dst_buffer = (
                            red_op.src_buffers,
                            red_op.dst_buffer,
                        )
                        for src_buffer in src_buffers:
                            assert (
                                len(self.buffers[d][dst_buffer.buffer_type])
                                > dst_buffer.index
                            )
                            self.buffers[d][dst_buffer.buffer_type][
                                dst_buffer.index
                            ] = self.buffers[d][src_buffer.buffer_type][
                                src_buffer.index
                            ]
                    logger.info(
                        "## Executed ## Device {}: {}".format(d, instr)
                    )
                    # increment instr_ptr
                    instr_ptr_per_device[d] += 1
                elif isinstance(instr, CommLaunchInstr):
                    is_comm_op[d] = True
                elif isinstance(instr, CommWaitInstr):
                    logger.info(
                        "## Executed ## Device {}: {}".format(d, instr)
                    )
                    # increment instr_ptr
                    instr_ptr_per_device[d] += 1
                else:
                    raise ValueError(
                        f"Unknown instruction type: {type(instr)}"
                    )
            # check if communication operations are matched
            if all(is_comm_op.values()):
                logger.info("Enter comm stage...")
                # comm stage
                # first check if all comm ops are matched
                instr_per_device = {
                    d: exec_plan.instructions[instr_ptr_per_device[d]]
                    for d, exec_plan in self.execution_plans.items()
                }
                is_matched, matchings, matched_key = (
                    find_matching_communications(
                        instr_per_device, exec_plan.buffer_info
                    )
                )
                assert is_matched, "Communication operations are not matched"
                instr_per_device_with_key = {
                    d: instr
                    for d, instr in instr_per_device.items()
                    if instr.key == matched_key
                }
                # simulate the communication op
                for d, instr in instr_per_device_with_key.items():
                    instr: CommLaunchInstr
                    for comm_op_idx, comm_op in enumerate(instr.comm_ops):
                        if comm_op.comm_type == CommType.RECV:
                            continue
                        peer_instr: CommLaunchInstr = (
                            instr_per_device_with_key[comm_op.peer]
                        )
                        recv_op: CommOp = peer_instr.comm_ops[
                            matchings[d][comm_op_idx]
                        ]
                        # check they match
                        assert is_matching_comm_op(
                            d,
                            comm_op,
                            comm_op.peer,
                            recv_op,
                            exec_plan.buffer_info,
                        )
                        send_buffer = self.buffers[d][
                            comm_op.buffer_block.buffer_type
                        ][comm_op.buffer_block.index]
                        assert (
                            len(
                                self.buffers[comm_op.peer][
                                    recv_op.buffer_block.buffer_type
                                ]
                            )
                            > recv_op.buffer_block.index
                        ), f"Recv buffer is not available for {recv_op}. Recv buffer: {self.buffers[comm_op.peer][recv_op.buffer_block.buffer_type]}"
                        assert (
                            send_buffer is not None
                        ), f"Send buffer is None for Device {d}: {comm_op}. Send buffer: {self.buffers[d][comm_op.buffer_block.buffer_type]}"
                        assert (
                            exec_plan.buffer_info[
                                comm_op.buffer_block.buffer_type
                            ].block_size
                            == exec_plan.buffer_info[
                                recv_op.buffer_block.buffer_type
                            ].block_size
                        )
                        assert (
                            exec_plan.buffer_info[
                                comm_op.buffer_block.buffer_type
                            ].block_numel
                            == exec_plan.buffer_info[
                                recv_op.buffer_block.buffer_type
                            ].block_numel
                        )
                        assert (
                            exec_plan.buffer_info[
                                comm_op.buffer_block.buffer_type
                            ].dtype
                            == exec_plan.buffer_info[
                                recv_op.buffer_block.buffer_type
                            ].dtype
                        )
                        logger.info(
                            "Executing Comm: {} -> {}".format(comm_op, recv_op)
                        )
                        self.buffers[comm_op.peer][
                            recv_op.buffer_block.buffer_type
                        ][recv_op.buffer_block.index] = send_buffer
                # increment instr_ptr for all devices
                for d in instr_per_device_with_key.keys():
                    instr_ptr_per_device[d] += 1


@pytest.mark.parametrize("n_nodes", [1, 2, 4])
@pytest.mark.parametrize("block_size", [32, 64])
@pytest.mark.parametrize("generate_backward", [False, True])
@pytest.mark.parametrize("attn_mask_type", ["causal", "two_range"])
def test_compiler(
    n_nodes: int,
    block_size: int,
    head_block_size: int,
    generate_backward: bool,
    attn_mask_type="causal",
):
    n_devices = 16
    seqlens = [1024, 2048, 4096, 8192]
    cu_seqlens = np.cumsum([0] + seqlens).tolist()
    qkv = torch.randn(sum(seqlens), 3, 12, 128, dtype=torch.bfloat16)
    if attn_mask_type == "causal":
        attn_mask = torch.zeros(sum(seqlens), 2, dtype=torch.int32)
        for seq_id, seqlen in enumerate(seqlens):
            for i in range(seqlen):
                attn_mask[cu_seqlens[seq_id] + i, 0] = 0
                attn_mask[cu_seqlens[seq_id] + i, 1] = i + 1
    elif attn_mask_type == "random":
        attn_mask = torch.zeros(sum(seqlens), 2, 2, dtype=torch.int32)
        for seq_id, seqlen in enumerate(seqlens):
            for i in range(seqlen):
                # 0 -> 128, i - 128 -> i
                attn_mask[cu_seqlens[seq_id] + i, 0, 0] = 0
                attn_mask[cu_seqlens[seq_id] + i, 0, 1] = min(128, i + 1)
                attn_mask[cu_seqlens[seq_id] + i, 1, 0] = max(0, i - 128)
                attn_mask[cu_seqlens[seq_id] + i, 1, 1] = i + 1
    else:
        raise ValueError(f"Unknown attn_mask_type: {attn_mask_type}")
    workload_spec = generate_workload(
        qkv,
        seqlens,
        block_size,
        head_block_size,
        n_devices,
        n_devices // n_nodes,
        comp_cost_model=AttnRooflineCostModel(),
        attn_mask=attn_mask,
    )
    exec_context = ExecutionContext(
        n_devices_per_node=n_devices // n_nodes,
        n_nodes=n_nodes,
        comm_cost_model=CommunicationCostModel(),
        comp_cost_model=AttnRooflineCostModel(),
    )
    compiler_config = CompilerConfig(
        mem_imbalance_epsilon=0.2,
        comp_imbalance_epsilon=0.2,
    )
    compiler = InstrCompiler(
        exec_context,
        compiler_config,
    )
    (
        fw_workload_spec,
        bw_workload,
        fw_execution_plan_map,
        bw_execution_plan_map,
    ) = compiler.compile(workload_spec, generate_backward=generate_backward)
    executor = DummyExecutor(
        fw_workload_spec,
        fw_execution_plan_map,
    )
    executor.execute()
    logger.info("Finished executing forward pass...")
    if generate_backward:
        executor = DummyExecutor(
            bw_workload,
            bw_execution_plan_map,
        )
        executor.execute()
        logger.info("Finished executing backward pass...")


if __name__ == "__main__":
    # pytest.main([__file__])
    test_compiler(2, 512, 1, False)
