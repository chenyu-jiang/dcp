from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch

from dcp.core.instructions import DType
from dcp.core.serialization import (
    SerializationConfig,
    bytes_to_int_meta,
    deserialize_dict_of_int_to_int,
    deserialize_dict_of_int_to_list_of_tuples,
    deserialize_dict_of_str_to_int,
    deserialize_dict_of_tuple_int_to_dict_of_int_to_int,
    deserialize_list_of_floats,
    deserialize_list_of_ints,
    deserialize_list_of_list_of_ints,
    deserialize_list_of_list_of_list_of_ints,
    deserialize_str,
    int_meta_to_bytes,
    serialize_dict_of_int_to_int,
    serialize_dict_of_int_to_list_of_ints,
    serialize_dict_of_str_to_int,
    serialize_dict_of_tuple_int_to_dict_of_int_to_int,
    serialize_list_of_floats,
    serialize_list_of_ints,
    serialize_list_of_list_of_ints,
    serialize_list_of_list_of_list_of_ints,
    serialize_str,
)


class BlockMeta:
    def __init__(
        self, type: str, dtype: DType, seq_id: int, head_id: int, **kwargs
    ):
        self.type = type
        self.dtype = dtype
        self.seq_id = seq_id
        self.head_id = head_id
        for k, v in kwargs.items():
            setattr(self, k, v)


class ComputationBlockMeta(BlockMeta):
    def __init__(
        self,
        type: str,
        dtype: DType,
        seq_id: int,
        head_id: int,
        q_id: int,
        kv_id: int,
        out_id: int,
        lse_id: int,
        dout_id: Optional[int] = None,
        dq_id: Optional[int] = None,
        dkv_id: Optional[int] = None,
        local_attn_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        super().__init__(type, dtype, seq_id, head_id, **kwargs)
        self.q_id = q_id
        self.kv_id = kv_id
        self.out_id = out_id
        self.lse_id = lse_id
        self.dout_id = dout_id
        self.dq_id = dq_id
        self.dkv_id = dkv_id
        if local_attn_mask is not None:
            self.local_attn_mask = local_attn_mask.to(torch.int32)
        else:
            self.local_attn_mask = None

    def __eq__(self, other):
        equal = (
            self.type == other.type
            and self.dtype == other.dtype
            and self.seq_id == other.seq_id
            and self.head_id == other.head_id
            and self.q_id == other.q_id
            and self.kv_id == other.kv_id
            and self.out_id == other.out_id
            and self.lse_id == other.lse_id
            and self.dout_id == other.dout_id
            and self.dq_id == other.dq_id
            and self.dkv_id == other.dkv_id
        )
        if (
            self.local_attn_mask is not None
            and other.local_attn_mask is not None
        ):
            equal &= torch.equal(self.local_attn_mask, other.local_attn_mask)
        return equal

    def serialize(self, config=SerializationConfig()) -> bytes:
        return b"".join(
            [
                serialize_str(self.type, config),
                (
                    self.dtype.serialize(config)
                    if isinstance(self.dtype, DType)
                    else int_meta_to_bytes(self.dtype, config)
                ),
                serialize_list_of_ints(
                    [
                        self.seq_id,
                        self.head_id,
                        self.q_id,
                        self.kv_id,
                        self.out_id,
                        self.lse_id,
                        self.dout_id,
                        self.dq_id,
                        self.dkv_id,
                    ]
                    if self.dout_id is not None
                    else [
                        self.seq_id,
                        self.head_id,
                        self.q_id,
                        self.kv_id,
                        self.out_id,
                        self.lse_id,
                    ]
                ),
                # serialize_tensor(self.local_attn_mask, config),
            ]
        )

    @classmethod
    def deserialize(cls, data: bytes, config=SerializationConfig()):
        type, data = deserialize_str(data, config)
        dtype, data = DType.deserialize(data, config)
        deserialized_ids, data = deserialize_list_of_ints(data)
        if len(deserialized_ids) == 6:
            (
                seq_id,
                head_id,
                q_id,
                kv_id,
                out_id,
                lse_id,
            ) = deserialized_ids
            dout_id = None
            dq_id = None
            dkv_id = None
        else:
            (
                seq_id,
                head_id,
                q_id,
                kv_id,
                out_id,
                lse_id,
                dout_id,
                dq_id,
                dkv_id,
            ) = deserialized_ids
        # local_attn_mask, data = deserialize_tensor(data, config)
        local_attn_mask = None
        return (
            cls(
                type,
                dtype,
                seq_id,
                head_id,
                q_id,
                kv_id,
                out_id,
                lse_id,
                dout_id,
                dq_id,
                dkv_id,
                local_attn_mask,
            ),
            data,
        )


class DataBlockMeta(BlockMeta):
    def __init__(
        self,
        type: str,
        dtype: DType,
        seq_id: int,
        head_id: int,
        block_id: int,
        n_tokens: int,  # actual number of tokens in the block
        block_size: int,  # in number of tokens
        head_block_size: int,  # in number of heads
        per_token_shape: Tuple[int, ...],
        **kwargs,
    ):
        super().__init__(type, dtype, seq_id, head_id, **kwargs)
        self.block_id = block_id
        self.n_tokens = n_tokens
        self.block_size = block_size
        self.head_block_size = head_block_size
        self.per_token_shape = per_token_shape

    def __eq__(self, value):
        return (
            self.type == value.type
            and self.dtype == value.dtype
            and self.seq_id == value.seq_id
            and self.head_id == value.head_id
            and self.block_id == value.block_id
            and self.n_tokens == value.n_tokens
            and self.block_size == value.block_size
            and self.head_block_size == value.head_block_size
            and self.per_token_shape == value.per_token_shape
        )

    def __repr__(self):
        return "DataBlockMeta(type={}, dtype={}, seq_id={}, head_id={}, block_id={}, n_tokens={}, block_size={}, head_block_size={}, per_token_shape={})".format(
            self.type,
            self.dtype,
            self.seq_id,
            self.head_id,
            self.block_id,
            self.n_tokens,
            self.block_size,
            self.head_block_size,
            self.per_token_shape,
        )

    def numel(self, padded=True):
        per_token_numel = 1
        for dim in self.per_token_shape:
            per_token_numel *= dim
        if padded:
            return self.block_size * per_token_numel
        return self.n_tokens * per_token_numel

    def serialize(self, config=SerializationConfig()) -> bytes:
        return b"".join(
            [
                serialize_str(self.type, config),
                (
                    self.dtype.serialize(config)
                    if isinstance(self.dtype, DType)
                    else int_meta_to_bytes(self.dtype, config)
                ),
                serialize_list_of_ints(
                    [
                        self.seq_id,
                        self.head_id,
                        self.block_id,
                        self.n_tokens,
                        self.block_size,
                        self.head_block_size,
                    ]
                ),
                serialize_list_of_ints(self.per_token_shape),
            ]
        )

    @classmethod
    def deserialize(cls, data: bytes, config=SerializationConfig()):
        type, data = deserialize_str(data, config)
        dtype, data = DType.deserialize(data, config)
        (
            seq_id,
            head_id,
            block_id,
            n_tokens,
            block_size,
            head_block_size,
        ), data = deserialize_list_of_ints(data)
        per_token_shape, data = deserialize_list_of_ints(data)
        per_token_shape = tuple(per_token_shape)
        return (
            cls(
                type,
                dtype,
                seq_id,
                head_id,
                block_id,
                n_tokens,
                block_size,
                head_block_size,
                per_token_shape,
            ),
            data,
        )


class BlockType:
    Q: str = "Q"
    dQ: str = "dQ"
    KV: str = "KV"
    dKV: str = "dKV"
    Out: str = "OUT"
    dOut: str = "dOUT"
    LSE: str = "LSE"
    Work: str = "WORK"


BlockMapping = Dict[int, BlockMeta]
DataBlockMapping = Dict[int, DataBlockMeta]
ComputationBlockMapping = Dict[int, ComputationBlockMeta]


def serialize_data_block_mapping(
    mapping: DataBlockMapping, config=SerializationConfig()
) -> bytes:
    serialized_mapping = b""
    serialized_mapping += int_meta_to_bytes(len(mapping), config)
    for key, meta in mapping.items():
        serialized_mapping += int_meta_to_bytes(key, config)
        serialized_mapping += meta.serialize(config)

    return serialized_mapping


def deserialize_data_block_mapping(
    data: bytes, config=SerializationConfig()
) -> Tuple[DataBlockMapping, bytes]:
    mapping = {}
    n_keys, data = bytes_to_int_meta(data, config)
    for _ in range(n_keys):
        key, data = bytes_to_int_meta(data, config)
        meta, data = DataBlockMeta.deserialize(data, config)
        mapping[key] = meta

    return mapping, data


def serialize_computation_block_mapping(
    mapping: ComputationBlockMapping, config=SerializationConfig()
) -> bytes:
    serialized_mapping = b""
    serialized_mapping += int_meta_to_bytes(len(mapping), config)
    for key, meta in mapping.items():
        serialized_mapping += int_meta_to_bytes(key, config)
        serialized_mapping += meta.serialize(config)

    return serialized_mapping


def deserialize_computation_block_mapping(
    data: bytes, config=SerializationConfig()
) -> Tuple[ComputationBlockMapping, bytes]:
    mapping = {}
    n_keys, data = bytes_to_int_meta(data, config)
    for _ in range(n_keys):
        key, data = bytes_to_int_meta(data, config)
        meta, data = ComputationBlockMeta.deserialize(data, config)
        mapping[key] = meta

    return mapping, data


@dataclass
class BlockMappings:
    input_id_to_meta: DataBlockMapping
    output_id_to_meta: DataBlockMapping
    work_id_to_meta: ComputationBlockMapping
    # device -> input_id -> buffer_index
    input_id_to_buffer_index: Dict[Tuple[int, int], Dict[int, int]]
    # device -> output_id -> buffer_index
    output_id_to_buffer_index: Dict[Tuple[int, int], Dict[int, int]]
    buffer_type_to_dtype: Dict[str, int]

    def serialize(self, config=SerializationConfig()) -> bytes:
        serialized_input_id_to_meta = serialize_data_block_mapping(
            self.input_id_to_meta, config
        )
        serialized_output_id_to_meta = serialize_data_block_mapping(
            self.output_id_to_meta, config
        )
        serialized_work_id_to_meta = serialize_computation_block_mapping(
            self.work_id_to_meta, config
        )
        serialized_input_id_to_buffer_index = (
            serialize_dict_of_tuple_int_to_dict_of_int_to_int(
                self.input_id_to_buffer_index, config
            )
        )
        serialized_output_id_to_buffer_index = (
            serialize_dict_of_tuple_int_to_dict_of_int_to_int(
                self.output_id_to_buffer_index, config
            )
        )
        serialized_buffer_type_to_dtype = serialize_dict_of_str_to_int(
            self.buffer_type_to_dtype, config
        )
        return b"".join(
            [
                serialized_input_id_to_meta,
                serialized_output_id_to_meta,
                serialized_work_id_to_meta,
                serialized_input_id_to_buffer_index,
                serialized_output_id_to_buffer_index,
                serialized_buffer_type_to_dtype,
            ]
        )

    @classmethod
    def deserialize(cls, data: bytes, config=SerializationConfig()):
        input_id_to_meta, data = deserialize_data_block_mapping(data, config)
        output_id_to_meta, data = deserialize_data_block_mapping(data, config)
        work_id_to_meta, data = deserialize_computation_block_mapping(
            data, config
        )
        input_id_to_buffer_index, data = (
            deserialize_dict_of_tuple_int_to_dict_of_int_to_int(data, config)
        )
        output_id_to_buffer_index, data = (
            deserialize_dict_of_tuple_int_to_dict_of_int_to_int(data, config)
        )
        buffer_type_to_dtype, data = deserialize_dict_of_str_to_int(
            data, config
        )
        return (
            cls(
                input_id_to_meta,
                output_id_to_meta,
                work_id_to_meta,
                input_id_to_buffer_index,
                output_id_to_buffer_index,
                buffer_type_to_dtype,
            ),
            data,
        )


class BlockManager:
    def __init__(self):
        self.blocks = []  # List[BlockId]
        self.data_index_map = {}  # data_id -> block_index

    def alloc_block(
        self,
        data_id: int,
        is_input: bool,
        creation_device: Tuple[int, int],
        stage_id: int,
        replica_id: int = 0,
    ):
        # first check if the block is already allocated
        block_key = (data_id, is_input, creation_device, stage_id, replica_id)
        if block_key in self.data_index_map:
            raise RuntimeError(
                f"block {data_id}, {'input' if is_input else 'output'} "
                f"creation device {creation_device}, "
                f"stage {stage_id}, replica {replica_id} is already allocated"
            )
        # check if empty block is available
        for i in range(len(self.blocks)):
            if self.blocks[i] is None:
                self.blocks[i] = block_key
                self.data_index_map[block_key] = i
                return i
        # no empty block, append to the end
        self.blocks.append(block_key)
        block_idx = len(self.blocks) - 1
        self.data_index_map[block_key] = block_idx
        return block_idx

    def free_block(
        self,
        data_id: int,
        is_input: bool,
        creation_device: Tuple[int, int],
        stage_id: int,
        replica_id: int = 0,
    ):
        block_key = (data_id, is_input, creation_device, stage_id, replica_id)
        block_idx = self.data_index_map[block_key]
        if block_idx is None:
            raise RuntimeError(
                f"block {data_id}, {'input' if is_input else 'output'} "
                f"creation device {creation_device}, "
                f"stage {stage_id}, replica {replica_id} is not allocated"
            )
        self.blocks[block_idx] = None
        del self.data_index_map[block_key]

    def free_block_by_idx(self, block_idx: int):
        block_key = self.blocks[block_idx]
        if block_key is None:
            raise RuntimeError(f"block {block_idx} is not allocated")
        del self.data_index_map[block_key]
        self.blocks[block_idx] = None

    def has_block(
        self,
        data_id: int,
        is_input: bool,
        creation_device: Optional[Tuple[int, int]] = None,
    ):
        if creation_device is None:
            return any(
                (data_id == k[0] and is_input == k[1])
                for k in self.data_index_map
            )
        return any(
            (data_id == k[0] and is_input == k[1] and creation_device == k[2])
            for k in self.data_index_map
        )

    def get_block(
        self,
        data_id: int,
        is_input: bool,
        creation_device: Tuple[int, int],
        stage_id: int,
        replica_id: int = 0,
    ):
        return self.data_index_map[
            (data_id, is_input, creation_device, stage_id, replica_id)
        ]

    def get_block_all_versions(
        self,
        data_id: int,
        is_input: bool,
        creation_device: Optional[Tuple[int, int]] = None,
    ):
        if creation_device is None:
            return [
                (index, k[2], k[3], k[4])
                for k, index in self.data_index_map.items()
                if data_id == k[0] and is_input == k[1]
            ]
        return [
            (index, k[3], k[4])
            for k, index in self.data_index_map.items()
            if data_id == k[0] and is_input == k[1] and creation_device == k[2]
        ]


@dataclass
class WorkloadSpec:
    workloads: List[float]
    work_unit_input_map: List[List[int]]
    work_unit_output_map: List[List[int]]
    block_mapping: BlockMappings
    input_to_device_map: Dict[int, Tuple[int, int]]
    output_to_device_map: Dict[int, Tuple[int, int]]
    work_to_device_map: Dict[int, Tuple[int, int]]
    work_to_stage_map: Dict[int, int]
    colocation_constraints: List[List[List[int]]]

    def __post_init__(self):
        self.input_sizes = []
        for i in range(len(self.block_mapping.input_id_to_meta)):
            meta = self.block_mapping.input_id_to_meta[i]
            self.input_sizes.append(
                meta.numel(padded=False) * DType.element_size(meta.dtype)
            )
        self.output_sizes = []
        for i in range(len(self.block_mapping.output_id_to_meta)):
            meta = self.block_mapping.output_id_to_meta[i]
            self.output_sizes.append(
                meta.numel(padded=False) * DType.element_size(meta.dtype)
            )

    def serialize(self, config=SerializationConfig()) -> bytes:
        return b"".join(
            [
                serialize_list_of_floats(self.workloads, config),
                serialize_list_of_list_of_ints(
                    self.work_unit_input_map, config
                ),
                serialize_list_of_list_of_ints(
                    self.work_unit_output_map, config
                ),
                self.block_mapping.serialize(config),
                serialize_dict_of_int_to_list_of_ints(
                    self.input_to_device_map, config
                ),
                serialize_dict_of_int_to_list_of_ints(
                    self.output_to_device_map, config
                ),
                serialize_dict_of_int_to_list_of_ints(
                    self.work_to_device_map, config
                ),
                serialize_dict_of_int_to_int(self.work_to_stage_map, config),
                serialize_list_of_list_of_list_of_ints(
                    self.colocation_constraints, config
                ),
            ]
        )

    @classmethod
    def deserialize(cls, data: bytes, config=SerializationConfig()):
        workloads, data = deserialize_list_of_floats(data, config)
        work_unit_input_map, data = deserialize_list_of_list_of_ints(
            data, config
        )
        work_unit_output_map, data = deserialize_list_of_list_of_ints(
            data, config
        )
        block_mapping, data = BlockMappings.deserialize(data, config)
        input_to_device_map, data = deserialize_dict_of_int_to_list_of_tuples(
            data, config
        )
        output_to_device_map, data = deserialize_dict_of_int_to_list_of_tuples(
            data, config
        )
        work_to_device_map, data = deserialize_dict_of_int_to_list_of_tuples(
            data, config
        )
        work_to_stage_map, data = deserialize_dict_of_int_to_int(data, config)
        colocation_constraints, data = (
            deserialize_list_of_list_of_list_of_ints(data, config)
        )

        return (
            cls(
                workloads,
                work_unit_input_map,
                work_unit_output_map,
                block_mapping,
                input_to_device_map,
                output_to_device_map,
                work_to_device_map,
                work_to_stage_map,
                colocation_constraints,
            ),
            data,
        )


def serialize_list_of_workloads(
    workloads: List[WorkloadSpec], config=SerializationConfig()
) -> bytes:
    return int_meta_to_bytes(len(workloads), config) + b"".join(
        [workload.serialize(config) for workload in workloads]
    )


def deserialize_list_of_workloads(
    data: bytes, config=SerializationConfig()
) -> Tuple[List[WorkloadSpec], bytes]:
    n_workloads, data = bytes_to_int_meta(data, config)
    workloads = []
    for _ in range(n_workloads):
        workload, data = WorkloadSpec.deserialize(data, config)
        workloads.append(workload)

    return workloads, data
