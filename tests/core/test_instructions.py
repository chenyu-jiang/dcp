# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Union

import numpy as np
import pytest

from dcp.core.instructions import *  # noqa: F403


def _test_serialization(instr: Union[BlockInstrBase, ExecutionPlan]):
    serialized = instr.serialize()
    assert isinstance(serialized, bytes)
    if isinstance(instr, BlockInstrBase):
        deserialized, remaining_bytes = BlockInstrBase.deserialize(serialized)
        assert len(remaining_bytes) == 0
    else:
        deserialized, _ = ExecutionPlan.deserialize(serialized)
    assert instr == deserialized
    # test casting to str
    serialized_casted = (
        serialized.decode("iso-8859-1").encode().decode().encode("iso-8859-1")
    )
    if isinstance(instr, BlockInstrBase):
        deserialized_casted, remaining_bytes = BlockInstrBase.deserialize(
            serialized_casted
        )
        assert len(remaining_bytes) == 0
    else:
        deserialized_casted, _ = ExecutionPlan.deserialize(serialized_casted)
    assert instr == deserialized_casted


all_buffer_types = [
    BufferType.LOCAL_KV,
    BufferType.LOCAL_Q,
    BufferType.LOCAL_OUT,
    BufferType.LOCAL_LSE,
    BufferType.LOCAL_dOUT,
    BufferType.BUFFER_Q,
    BufferType.BUFFER_KV,
    BufferType.BUFFER_LSE,
]


def _get_memcpy_instr(n_inputs: int, buffer_type: str):
    src_dst_pairs = []
    for i in range(n_inputs):
        n_tokens = int(np.random.randint(1, 100))
        src_buffer = BufferBlock(buffer_type, i, n_tokens=n_tokens)
        dst_buffer = BufferBlock(buffer_type, i + 1, n_tokens=n_tokens)
        src_dst_pairs.append((src_buffer, dst_buffer))
    return MemcpyInstr(src_dst_pairs)


@pytest.mark.parametrize("n_inputs", [1, 2, 4])
@pytest.mark.parametrize("buffer_type", all_buffer_types)
def test_serialization_memcpy(
    n_inputs: int,
    buffer_type: str,
):
    instr = _get_memcpy_instr(n_inputs, buffer_type)
    _test_serialization(instr)


def _get_comm_launch_instr(n_inputs: int, buffer_type: str):
    comm_ops = []
    for i in range(n_inputs):
        comm_ops.append(
            CommOp(
                comm_type=CommType.SEND if i % 2 == 0 else CommType.RECV,
                peer=(i, i + 1),
                buffer_block=BufferBlock(buffer_type, i, n_tokens=100),
            )
        )
    return CommLaunchInstr("comm_id", comm_ops, stream="stream")


@pytest.mark.parametrize("n_inputs", [1, 2, 4])
@pytest.mark.parametrize("buffer_type", all_buffer_types)
def test_serialization_comm_launch_instr(
    n_inputs: int,
    buffer_type: str,
):
    instr = _get_comm_launch_instr(n_inputs, buffer_type)
    _test_serialization(instr)


def _get_comm_wait_instr(key: str):
    return CommWaitInstr(key, stream="stream")


def test_serialization_comm_wait_instr():
    instr = _get_comm_wait_instr("comm_id")
    _test_serialization(instr)


def _get_attn_instr(seed: int, attn_mask_dim: int, fwd: bool = True):
    rng = np.random.default_rng(seed)
    # generate random stage_id
    stage_id = int(rng.integers(0, 100))
    seqlens_q = rng.integers(2, 100, size=rng.integers(2, 10)).tolist()
    seqlens_kv = rng.integers(2, 100, size=rng.integers(2, 10)).tolist()
    max_seqlen_q = max(seqlens_q)
    max_seqlen_kv = max(seqlens_kv)
    n_seqs = rng.integers(1, 10)

    def _get_list_of_buffers(length: int):
        buffer_indices = rng.integers(0, max_seqlen_kv, size=length).tolist()
        buffers = [
            all_buffer_types[x]
            for x in rng.integers(
                0, len(all_buffer_types), size=length
            ).tolist()
        ]
        return [
            BufferBlock(buffer, buffer_index, n_tokens=100)
            for buffer, buffer_index in zip(buffers, buffer_indices)
        ]

    # generate random list
    def _get_block_table():
        block_table = []
        for _ in range(n_seqs):
            length = rng.integers(1, len(seqlens_kv))
            block_table.append(_get_list_of_buffers(length))
        return block_table

    def _get_attn_mask(dim: int):
        if dim == 3:
            return [
                torch.tensor(
                    rng.integers(0, max_seqlen_kv, size=(2, sum(seqlens_q))),
                    dtype=torch.int32,
                ),
                torch.tensor(
                    rng.integers(0, max_seqlen_kv, size=(2, sum(seqlens_q))),
                    dtype=torch.int32,
                ),
            ]
        else:
            return [
                torch.tensor(
                    rng.integers(0, max_seqlen_kv, size=(2, sum(seqlens_q))),
                    dtype=torch.int32,
                )
            ]

    if fwd:
        return AttnInstr(
            stage_id,
            seqlens_q,
            seqlens_kv,
            max_seqlen_q,
            max_seqlen_kv,
            _get_attn_mask(attn_mask_dim),
            _get_block_table(),
            _get_block_table(),
            _get_block_table(),
            _get_block_table(),
        )
    else:
        return AttnBackwardInstr(
            stage_id,
            seqlens_q,
            seqlens_kv,
            max_seqlen_q,
            max_seqlen_kv,
            _get_attn_mask(attn_mask_dim),
            _get_block_table(),
            _get_block_table(),
            _get_block_table(),
            _get_block_table(),
            _get_block_table(),
        )


@pytest.mark.parametrize("seed", [0, 42, 1000])
@pytest.mark.parametrize("attn_mask_dim", [2, 3])
@pytest.mark.parametrize("fwd", [True, False])
def test_serialization_attn_instr(seed: int, attn_mask_dim: int, fwd: bool):
    instr = _get_attn_instr(seed, attn_mask_dim, fwd)
    _test_serialization(instr)


def _get_reduction_instr(n_inputs: int, n_buffer_per_op: int):
    red_ops = []
    for i in range(n_inputs):
        src_buffers = []
        for j in range(n_buffer_per_op):
            src_buffers.append(
                (
                    BufferBlock(BufferType.BUFFER_Q, j, n_tokens=100),
                    BufferBlock(BufferType.BUFFER_LSE, j, n_tokens=100),
                )
            )
        dst_buffer = (
            BufferBlock(BufferType.LOCAL_OUT, i, n_tokens=100),
            BufferBlock(BufferType.LOCAL_LSE, i, n_tokens=100),
        )
        red_ops.append(ReductionOp(src_buffers, dst_buffer))
    return AttnReductionInstr(red_ops, True)


@pytest.mark.parametrize("n_inputs", [1, 2, 4])
@pytest.mark.parametrize("n_buffer_per_op", [1, 2, 4])
def test_serialization_reduction_instr(n_inputs, n_buffer_per_op):
    instr = _get_reduction_instr(n_inputs, n_buffer_per_op)
    _test_serialization(instr)


def _get_sum_instr(n_inputs: int, n_buffer_per_op: int):
    sum_ops = []
    for i in range(n_inputs):
        src_buffers = []
        for j in range(n_buffer_per_op):
            src_buffers.append(
                BufferBlock(BufferType.BUFFER_Q, j, n_tokens=100)
            )
        dst_buffer = BufferBlock(BufferType.LOCAL_OUT, i, n_tokens=100)
        sum_ops.append(SumOp(src_buffers, dst_buffer))
    return SumInstr(sum_ops, True)


@pytest.mark.parametrize("n_inputs", [1, 2, 4])
@pytest.mark.parametrize("n_buffer_per_op", [1, 2, 4])
def test_serialization_sum_instr(n_inputs, n_buffer_per_op):
    instr = _get_sum_instr(n_inputs, n_buffer_per_op)
    _test_serialization(instr)


def test_serialization_exec_plan():
    instructions = [
        _get_comm_launch_instr(2, BufferType.LOCAL_KV),
        _get_comm_launch_instr(4, BufferType.LOCAL_Q),
        _get_comm_wait_instr("comm_id"),
        _get_attn_instr(42, 2),
        _get_attn_instr(24, 2, fwd=False),
        _get_attn_instr(42, 3),
        _get_attn_instr(24, 3, fwd=False),
        _get_memcpy_instr(4, BufferType.BUFFER_Q),
        _get_memcpy_instr(2, BufferType.LOCAL_KV),
        _get_comm_launch_instr(2, BufferType.BUFFER_KV),
        _get_comm_launch_instr(2, BufferType.BUFFER_LSE),
        _get_reduction_instr(2, 2),
        _get_sum_instr(2, 2),
    ]
    exec_plan = ExecutionPlan(
        instructions,
        n_nodes=2,
        n_devices_per_node=8,
        node_id=0,
        local_device_id=1,
        n_stages=4,
        local_cu_seqlens=[128, 256],
        buffer_info={
            BufferType.BUFFER_KV: BufferInfo(
                8, 4, 32, DType.BFLOAT16, (128, 128), (128, 256)
            ),
            BufferType.BUFFER_Q: BufferInfo(
                16, 8, 64, DType.BFLOAT16, (128, 256), (128, 128)
            ),
            BufferType.BUFFER_LSE: BufferInfo(
                8, 4, 16, DType.FLOAT32, (128, 128), (128, 256)
            ),
            BufferType.LOCAL_KV: BufferInfo(
                8, 4, 128, DType.BFLOAT16, (64, 128), (128, 128)
            ),
            BufferType.LOCAL_Q: BufferInfo(
                32, 8, 32, DType.FLOAT16, (128, 128), (64, 128)
            ),
        },
    )
    _test_serialization(exec_plan)


if __name__ == "__main__":
    pytest.main([__file__])
