from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Type

import torch

from dcp.core.serialization import (
    SerializationConfig,
    bytes_to_int_meta,
    deserialize_int_tuple,
    deserialize_list_of_ints,
    deserialize_str,
    int_meta_to_bytes,
    serialize_int_tuple,
    serialize_list_of_ints,
    serialize_str,
)


class BufferType:
    # local persistent buffers
    LOCAL_Q = "LocalQ"
    LOCAL_KV = "LocalKV"
    LOCAL_dQ = "LocaldQ"
    LOCAL_dKV = "LocaldKV"
    LOCAL_OUT = "LocalOut"
    LOCAL_dOUT = "LocaldOut"
    LOCAL_LSE = "LocalLSE"
    # temporary buffers
    BUFFER_Q = "BufferQ"
    BUFFER_KV = "BufferKV"
    BUFFER_dQ = "BufferdQ"
    BUFFER_dKV = "BufferdKV"
    BUFFER_OUT = "BufferOut"
    BUFFER_dOUT = "BufferdOut"
    BUFFER_LSE = "BufferLSE"


class CommType:
    SEND = 0
    RECV = 1

    @classmethod
    def from_num(cls, num):
        if num == 0:
            return cls.SEND
        elif num == 1:
            return cls.RECV
        else:
            raise ValueError(f"Invalid comm type {num}")

    @classmethod
    def to_str(cls, comm_type):
        if comm_type == cls.SEND:
            return "send"
        elif comm_type == cls.RECV:
            return "recv"
        else:
            raise ValueError(f"Invalid comm type {comm_type}")


class DType:
    BFLOAT16 = 0
    FLOAT16 = 1
    FLOAT32 = 2
    INT = 3

    @classmethod
    def from_num(cls, num):
        if num == 0:
            return cls.BFLOAT16
        elif num == 1:
            return cls.FLOAT16
        elif num == 2:
            return cls.FLOAT32
        elif num == 3:
            return cls.INT
        else:
            raise ValueError(f"Invalid dtype {num}")

    @classmethod
    def as_str(cls, dtype):
        if dtype == cls.BFLOAT16:
            return "bf16"
        elif dtype == cls.FLOAT16:
            return "fp16"
        elif dtype == cls.FLOAT32:
            return "fp32"
        elif dtype == cls.INT:
            return "int"
        else:
            raise ValueError(f"Invalid dtype {dtype}")

    @classmethod
    def element_size(cls, dtype):
        if dtype == cls.BFLOAT16:
            return 2
        elif dtype == cls.FLOAT16:
            return 2
        elif dtype == cls.FLOAT32:
            return 4
        elif dtype == cls.INT:
            return 4
        else:
            raise ValueError(f"Invalid dtype {dtype}")

    def serialize(self, config=SerializationConfig()) -> bytes:
        return int_meta_to_bytes(int(self), config)

    @classmethod
    def deserialize(
        cls, bytes: bytes, config=SerializationConfig()
    ) -> Tuple[DType, bytes]:
        dtype, bytes = bytes_to_int_meta(bytes, config)
        return DType.from_num(dtype), bytes


class BlockInstrBase:
    """Base class for all instructions to be executed by dcp.
    All keyword arguments are stored as members similar to a ``namedtuple``.
    These are then accessible to the PipeEngine during execution.
    Args:
        kwargs (optional): keyword arguments to store as members
    """

    # used to generate a unique index for each instruction class
    # for serialization
    _instr_index_to_cls: Dict[int, Type[BlockInstrBase]] = {}

    def __init__(self, **kwargs):
        self.name = self.__class__.__name__
        self.kwargs = kwargs
        for key, val in kwargs.items():
            setattr(self, key, val)

    def __repr__(self):
        name = f"{self.name}("
        if self.kwargs:
            name += ", ".join(
                f"{key}={repr(arg)}" for key, arg in self.kwargs.items()
            )
        name += ")"
        return name

    def __init_subclass__(cls) -> None:
        cls._instr_index = len(BlockInstrBase._instr_index_to_cls)
        BlockInstrBase._instr_index_to_cls[cls._instr_index] = cls

    def serialize(self, config=SerializationConfig()) -> bytes:
        """Serialize the instruction to a byte array."""
        return self._instr_index.to_bytes(
            config.INSTRUCTION_INDEX_BYTES, config.BYTES_ENDIANNESS
        )

    def _deserialize(
        bytes: bytes, config=SerializationConfig()
    ) -> Tuple[Dict[str, Any], bytes]:
        return {}, bytes

    @classmethod
    def deserialize(
        cls, bytes: bytes, config=SerializationConfig()
    ) -> Tuple[BlockInstrBase, bytes]:
        """Deserialize the instruction from a byte array."""
        instr_index = int.from_bytes(
            bytes[: config.INSTRUCTION_INDEX_BYTES], config.BYTES_ENDIANNESS
        )
        bytes = bytes[config.INSTRUCTION_INDEX_BYTES :]
        kwargs, bytes = cls._instr_index_to_cls[instr_index]._deserialize(
            bytes, config=config
        )
        return (
            cls._instr_index_to_cls[instr_index](**kwargs),
            bytes,
        )

    def __eq__(self, other: BlockInstrBase):
        return (
            self.__class__ == other.__class__ and self.kwargs == other.kwargs
        )


@dataclass(frozen=True)
class BufferBlock:
    buffer_type: str
    index: int
    n_tokens: int

    def __post_init__(self):
        assert self.buffer_type is not None
        assert self.index is not None
        assert isinstance(self.index, int)
        assert self.n_tokens is not None
        assert isinstance(self.n_tokens, int)

    def __repr__(self):
        return f"({self.buffer_type}, {self.index}, {self.n_tokens})"


def serialize_buffer_block(
    buffer_block: BufferBlock, config=SerializationConfig()
) -> bytes:
    return (
        serialize_str(buffer_block.buffer_type, config)
        + int_meta_to_bytes(buffer_block.index, config)
        + int_meta_to_bytes(buffer_block.n_tokens, config)
    )


def deserialize_buffer_block(
    bytes: bytes, config=SerializationConfig()
) -> Tuple[BufferBlock, bytes]:
    buffer_name, bytes = deserialize_str(bytes, config)
    buffer_index, bytes = bytes_to_int_meta(bytes, config)
    n_tokens, bytes = bytes_to_int_meta(bytes, config)
    return BufferBlock(buffer_name, buffer_index, n_tokens), bytes


def serialize_list_of_buffer_blocks(
    buffer_block_list: List[BufferBlock], config=SerializationConfig()
) -> bytes:
    return int_meta_to_bytes(len(buffer_block_list), config) + b"".join(
        [serialize_buffer_block(b, config) for b in buffer_block_list]
    )


def deserialize_list_of_buffer_blocks(
    bytes: bytes, config=SerializationConfig()
) -> Tuple[List[BufferBlock], bytes]:
    n_blocks, bytes = bytes_to_int_meta(bytes, config)
    buffer_blocks = []
    for _ in range(n_blocks):
        buffer_block, bytes = deserialize_buffer_block(bytes, config)
        buffer_blocks.append(buffer_block)
    return buffer_blocks, bytes


def _serialize_block_table(
    block_table: List[List[int]], config=SerializationConfig()
) -> bytes:
    return int_meta_to_bytes(len(block_table), config) + b"".join(
        [
            serialize_list_of_buffer_blocks(block_list, config)
            for block_list in block_table
        ]
    )


def _deserialize_block_table(
    bytes: bytes, config=SerializationConfig()
) -> Tuple[List[List[int]], bytes]:
    batch_size, bytes = bytes_to_int_meta(bytes, config)
    block_table = []
    for _ in range(batch_size):
        buffer_blocks, bytes = deserialize_list_of_buffer_blocks(bytes, config)
        block_table.append(buffer_blocks)
    return block_table, bytes


def _serialize_attn_mask(
    attn_mask: List[torch.Tensor],
    config=SerializationConfig(),
) -> bytes:
    res = int_meta_to_bytes(len(attn_mask), config)
    for mask in attn_mask:
        res += serialize_int_tuple(mask.shape, config)
        assert mask.dtype == torch.int32
        mask_bytes = mask.cpu().numpy().tobytes()
        res += int_meta_to_bytes(len(mask_bytes), config)
        res += mask_bytes
    return res


def _deserialize_attn_mask(
    bytes: bytes, config=SerializationConfig()
) -> Tuple[List[torch.Tensor], bytes]:
    n_masks, bytes = bytes_to_int_meta(bytes, config)
    attn_mask = []
    for _ in range(n_masks):
        mask_shape, bytes = deserialize_int_tuple(bytes, config)
        mask_len, bytes = bytes_to_int_meta(bytes, config)
        mask_bytes = bytes[:mask_len]
        bytes = bytes[mask_len:]
        mask = torch.frombuffer(
            bytearray(mask_bytes), dtype=torch.int32
        ).reshape(mask_shape)
        attn_mask.append(mask)
    return attn_mask, bytes


def _eq_attn_mask(
    mask1: List[torch.Tensor], mask2: List[torch.Tensor]
) -> bool:
    if len(mask1) != len(mask2):
        return False
    for m1, m2 in zip(mask1, mask2):
        if not torch.equal(m1, m2):
            return False
    return True


class MemcpyInstr(BlockInstrBase):
    def __init__(
        self,
        src_dst_pairs: List[Tuple[BufferBlock, BufferBlock]],
        **kwargs,
    ):
        super().__init__(
            src_dst_pairs=src_dst_pairs,
            **kwargs,
        )
        self.src_dst_pairs: List[Tuple[BufferBlock, BufferBlock]]

    def serialize(self, config=SerializationConfig()) -> bytes:
        return (
            super().serialize(config=config)
            + int_meta_to_bytes(len(self.src_dst_pairs), config)
            + b"".join(
                [
                    serialize_buffer_block(src, config)
                    + serialize_buffer_block(dst, config)
                    for src, dst in self.src_dst_pairs
                ]
            )
        )

    @classmethod
    def _deserialize(
        cls, bytes, config=SerializationConfig()
    ) -> Tuple[Dict[str, Any], bytes]:
        kwargs, bytes = super()._deserialize(bytes, config=config)
        n_pairs, bytes = bytes_to_int_meta(bytes, config)
        src_dst_pairs = []
        for _ in range(n_pairs):
            src, bytes = deserialize_buffer_block(bytes, config)
            dst, bytes = deserialize_buffer_block(bytes, config)
            src_dst_pairs.append((src, dst))
        kwargs.update(
            {
                "src_dst_pairs": src_dst_pairs,
            }
        )
        return kwargs, bytes


@dataclass(frozen=True)
class CommOp:
    comm_type: CommType
    peer: Tuple[int, int]
    buffer_block: BufferBlock

    def serialize(self, config=SerializationConfig()) -> bytes:
        return (
            int_meta_to_bytes(self.comm_type, config)
            + int_meta_to_bytes(self.peer[0], config)
            + int_meta_to_bytes(self.peer[1], config)
            + serialize_buffer_block(self.buffer_block, config)
        )

    @classmethod
    def _deserialize(
        cls, bytes, config=SerializationConfig()
    ) -> Tuple[Dict[str, Any], bytes]:
        comm_type, bytes = bytes_to_int_meta(bytes, config)
        comm_type = CommType.from_num(comm_type)
        node_id, bytes = bytes_to_int_meta(bytes, config)
        device_id, bytes = bytes_to_int_meta(bytes, config)
        buffer_block, bytes = deserialize_buffer_block(bytes, config)
        comm_op = CommOp(comm_type, (node_id, device_id), buffer_block)
        return comm_op, bytes

    def __repr__(self) -> str:
        return f"CommOp(comm_type={CommType.to_str(self.comm_type)}, peer={self.peer}, buffer_block={self.buffer_block})"


class CommLaunchInstr(BlockInstrBase):
    def __init__(
        self,
        key: str,
        comm_ops: List[CommOp],
        stream: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(
            key=key,
            comm_ops=comm_ops,
            stream=stream,
            **kwargs,
        )
        self.key: str
        self.comm_ops: List[CommOp]
        self.stream: Optional[str]

    def serialize(self, config=SerializationConfig()) -> bytes:
        return (
            super().serialize(config=config)
            + serialize_str(self.key, config)
            + len(self.comm_ops).to_bytes(
                config.EXECUTION_PLAN_META_BYTES, config.BYTES_ENDIANNESS
            )
            + b"".join(
                [comm_op.serialize(config) for comm_op in self.comm_ops]
            )
            + serialize_str(self.stream, config)
        )

    @classmethod
    def _deserialize(
        cls, bytes, config=SerializationConfig()
    ) -> Tuple[Dict[str, Any], bytes]:
        kwargs, bytes = super()._deserialize(bytes, config=config)
        key, bytes = deserialize_str(bytes, config)
        n_comm_ops = int.from_bytes(
            bytes[: config.EXECUTION_PLAN_META_BYTES], config.BYTES_ENDIANNESS
        )
        bytes = bytes[config.EXECUTION_PLAN_META_BYTES :]
        comm_ops = []
        for _ in range(n_comm_ops):
            comm_op, bytes = CommOp._deserialize(bytes, config=config)
            comm_ops.append(comm_op)
        stream, bytes = deserialize_str(bytes, config)
        if stream == "":
            stream = None
        kwargs.update(
            {
                "key": key,
                "comm_ops": comm_ops,
                "stream": stream,
            }
        )
        return kwargs, bytes


class BarrierInstr(BlockInstrBase):
    pass


class CommWaitInstr(BlockInstrBase):
    def __init__(
        self,
        key: str,
        stream: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(
            key=key,
            stream=stream,
            **kwargs,
        )
        self.key: str
        self.stream: Optional[str]

    def serialize(self, config=SerializationConfig()) -> bytes:
        return (
            super().serialize(config=config)
            + serialize_str(self.key, config)
            + serialize_str(self.stream, config)
        )

    @classmethod
    def _deserialize(
        cls, bytes, config=SerializationConfig()
    ) -> Tuple[Dict[str, Any], bytes]:
        kwargs, bytes = super()._deserialize(bytes, config=config)
        key, bytes = deserialize_str(bytes, config)
        stream, bytes = deserialize_str(bytes, config)
        if stream == "":
            stream = None
        kwargs.update(
            {
                "key": key,
                "stream": stream,
            }
        )
        return kwargs, bytes


def _repr_list_of_blocks(blocks: List[BufferBlock]) -> str:
    s = "["
    for i, block in enumerate(blocks):
        s += f"{block.index}"
        if i < len(blocks) - 1:
            s += ", "
    s += "]"
    return s


def _repr_block_table(block_table: List[List[BufferBlock]]) -> str:
    buffer_type = block_table[0][0].buffer_type
    s = f"{buffer_type}["
    for i, block_list in enumerate(block_table):
        s += _repr_list_of_blocks(block_list)
        if i < len(block_table) - 1:
            s += ", "
    s += "]"
    return s


class AttnInstr(BlockInstrBase):
    """
    Carry out attention computation for given blocks.
    """

    def __init__(
        self,
        stage_id: int,
        seqlens_q: List[int],
        seqlens_kv: List[int],
        max_seqlen_q: int,
        max_seqlen_kv: int,
        attn_mask: List[List[int]] | List[List[List[int]]],
        q_block_table: List[List[BufferBlock]],
        kv_block_table: List[List[BufferBlock]],
        out_block_table: List[List[BufferBlock]],
        lse_block_table: List[List[BufferBlock]],
        **kwargs,
    ):
        super().__init__(
            stage_id=stage_id,
            seqlens_q=seqlens_q,
            seqlens_kv=seqlens_kv,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_kv=max_seqlen_kv,
            attn_mask=attn_mask,
            q_block_table=q_block_table,
            kv_block_table=kv_block_table,
            out_block_table=out_block_table,
            lse_block_table=lse_block_table,
            **kwargs,
        )
        self.stage_id: int
        self.seqlens_q: List[int]
        self.seqlens_kv: List[int]
        self.max_seqlen_q: int
        self.max_seqlen_kv: int
        self.attn_mask: List[List[int]] | List[List[List[int]]]
        self.q_block_table: List[List[BufferBlock]]
        self.kv_block_table: List[List[BufferBlock]]
        self.out_block_table: List[List[BufferBlock]]
        self.lse_block_table: List[List[BufferBlock]]

    def __post_init__(self):
        # check out and lse block table are the same
        assert len(self.out_block_table) == len(self.lse_block_table)
        for dim0 in range(len(self.out_block_table)):
            assert len(self.out_block_table[dim0]) == len(
                self.lse_block_table[dim0]
            )
            for dim1 in range(len(self.out_block_table[dim0])):
                assert (
                    self.out_block_table[dim0][dim1].index
                    == self.lse_block_table[dim0][dim1].index
                )

    def __repr__(self) -> str:
        return (
            f"AttnInstr(stage_id={self.stage_id}, seqlens_q={self.seqlens_q}, "
            f"seqlens_kv={self.seqlens_kv}, max_seqlen_q={self.max_seqlen_q}, "
            f"max_seqlen_kv={self.max_seqlen_kv}, "
            # f"attn_mask={str(self.attn_mask)}, "
            f"q_block_table={_repr_block_table(self.q_block_table)}, "
            f"kv_block_table={_repr_block_table(self.kv_block_table)}, "
            f"out_block_table={_repr_block_table(self.out_block_table)}, "
            f"lse_block_table={_repr_block_table(self.lse_block_table)})"
        )

    def __eq__(self, other: AttnInstr) -> bool:
        return (
            self.stage_id == other.stage_id
            and self.seqlens_q == other.seqlens_q
            and self.seqlens_kv == other.seqlens_kv
            and self.max_seqlen_q == other.max_seqlen_q
            and self.max_seqlen_kv == other.max_seqlen_kv
            and _eq_attn_mask(self.attn_mask, other.attn_mask)
            and self.q_block_table == other.q_block_table
            and self.kv_block_table == other.kv_block_table
            and self.out_block_table == other.out_block_table
            and self.lse_block_table == other.lse_block_table
        )

    def serialize(self, config=SerializationConfig()) -> bytes:
        return (
            super().serialize(config=config)
            + int_meta_to_bytes(self.stage_id, config)
            + serialize_list_of_ints(self.seqlens_q, config)
            + serialize_list_of_ints(self.seqlens_kv, config)
            + int_meta_to_bytes(self.max_seqlen_q, config)
            + int_meta_to_bytes(self.max_seqlen_kv, config)
            + _serialize_attn_mask(self.attn_mask, config)
            + _serialize_block_table(self.q_block_table, config)
            + _serialize_block_table(self.kv_block_table, config)
            + _serialize_block_table(self.out_block_table, config)
            + _serialize_block_table(self.lse_block_table, config)
        )

    @classmethod
    def _deserialize(
        cls, bytes, config=SerializationConfig()
    ) -> Tuple[Dict[str, Any], bytes]:
        kwargs, bytes = super()._deserialize(bytes, config=config)
        stage_id, bytes = bytes_to_int_meta(bytes, config)
        seqlens_q, bytes = deserialize_list_of_ints(bytes, config)
        seqlens_kv, bytes = deserialize_list_of_ints(bytes, config)
        max_seqlen_q, bytes = bytes_to_int_meta(bytes, config)
        max_seqlen_kv, bytes = bytes_to_int_meta(bytes, config)
        attn_mask, bytes = _deserialize_attn_mask(bytes, config)
        q_block_table, bytes = _deserialize_block_table(bytes, config)
        kv_block_table, bytes = _deserialize_block_table(bytes, config)
        out_block_table, bytes = _deserialize_block_table(bytes, config)
        lse_block_table, bytes = _deserialize_block_table(bytes, config)
        kwargs.update(
            {
                "stage_id": stage_id,
                "seqlens_q": seqlens_q,
                "seqlens_kv": seqlens_kv,
                "max_seqlen_q": max_seqlen_q,
                "max_seqlen_kv": max_seqlen_kv,
                "attn_mask": attn_mask,
                "q_block_table": q_block_table,
                "kv_block_table": kv_block_table,
                "out_block_table": out_block_table,
                "lse_block_table": lse_block_table,
            }
        )
        return kwargs, bytes


class AttnBackwardInstr(BlockInstrBase):
    def __init__(
        self,
        stage_id: int,
        seqlens_q: List[int],
        seqlens_kv: List[int],
        max_seqlen_q: int,
        max_seqlen_kv: int,
        attn_mask: List[List[int]] | List[List[List[int]]],
        q_block_table: List[List[BufferBlock]],
        kv_block_table: List[List[BufferBlock]],
        out_block_table: List[List[BufferBlock]],
        dq_block_table: List[List[BufferBlock]],
        dkv_block_table: List[List[BufferBlock]],
        **kwargs,
    ):
        super().__init__(
            stage_id=stage_id,
            seqlens_q=seqlens_q,
            seqlens_kv=seqlens_kv,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_kv=max_seqlen_kv,
            attn_mask=attn_mask,
            q_block_table=q_block_table,
            kv_block_table=kv_block_table,
            out_block_table=out_block_table,
            dq_block_table=dq_block_table,
            dkv_block_table=dkv_block_table,
            **kwargs,
        )
        self.stage_id: int
        self.seqlens_q: List[int]
        self.seqlens_kv: List[int]
        self.max_seqlen_q: int
        self.max_seqlen_kv: int
        self.attn_mask: List[List[int]] | List[List[List[int]]]
        self.q_block_table: List[List[BufferBlock]]
        self.kv_block_table: List[List[BufferBlock]]
        self.out_block_table: List[List[BufferBlock]]
        self.dq_block_table: List[List[BufferBlock]]
        self.dkv_block_table: List[List[BufferBlock]]

    def __repr__(self) -> str:
        return (
            f"AttnBackwardInstr(stage_id={self.stage_id}, seqlens_q={self.seqlens_q}, "
            f"seqlens_kv={self.seqlens_kv}, max_seqlen_q={self.max_seqlen_q}, "
            f"max_seqlen_kv={self.max_seqlen_kv}, "
            # f"attn_mask={str(self.attn_mask)}, "
            f"q_block_table={_repr_block_table(self.q_block_table)}, "
            f"kv_block_table={_repr_block_table(self.kv_block_table)}, "
            f"out_block_table={_repr_block_table(self.out_block_table)}, "
            f"dq_block_table={_repr_block_table(self.dq_block_table)}, "
            f"dkv_block_table={_repr_block_table(self.dkv_block_table)})"
        )

    def __eq__(self, other: AttnBackwardInstr) -> bool:
        return (
            self.stage_id == other.stage_id
            and self.seqlens_q == other.seqlens_q
            and self.seqlens_kv == other.seqlens_kv
            and self.max_seqlen_q == other.max_seqlen_q
            and self.max_seqlen_kv == other.max_seqlen_kv
            and _eq_attn_mask(self.attn_mask, other.attn_mask)
            and self.q_block_table == other.q_block_table
            and self.kv_block_table == other.kv_block_table
            and self.out_block_table == other.out_block_table
            and self.dq_block_table == other.dq_block_table
            and self.dkv_block_table == other.dkv_block_table
        )

    def serialize(self, config=SerializationConfig()) -> bytes:
        return (
            super().serialize(config=config)
            + int_meta_to_bytes(self.stage_id, config)
            + serialize_list_of_ints(self.seqlens_q, config)
            + serialize_list_of_ints(self.seqlens_kv, config)
            + int_meta_to_bytes(self.max_seqlen_q, config)
            + int_meta_to_bytes(self.max_seqlen_kv, config)
            + _serialize_attn_mask(self.attn_mask, config)
            + _serialize_block_table(self.q_block_table, config)
            + _serialize_block_table(self.kv_block_table, config)
            + _serialize_block_table(self.out_block_table, config)
            + _serialize_block_table(self.dq_block_table, config)
            + _serialize_block_table(self.dkv_block_table, config)
        )

    @classmethod
    def _deserialize(
        cls, bytes, config=SerializationConfig()
    ) -> Tuple[Dict[str, Any], bytes]:
        kwargs, bytes = super()._deserialize(bytes, config=config)
        stage_id, bytes = bytes_to_int_meta(bytes, config)
        seqlens_q, bytes = deserialize_list_of_ints(bytes, config)
        seqlens_kv, bytes = deserialize_list_of_ints(bytes, config)
        max_seqlen_q, bytes = bytes_to_int_meta(bytes, config)
        max_seqlen_kv, bytes = bytes_to_int_meta(bytes, config)
        attn_mask, bytes = _deserialize_attn_mask(bytes, config)
        q_block_table, bytes = _deserialize_block_table(bytes, config)
        kv_block_table, bytes = _deserialize_block_table(bytes, config)
        out_block_table, bytes = _deserialize_block_table(bytes, config)
        dq_block_table, bytes = _deserialize_block_table(bytes, config)
        dkv_block_table, bytes = _deserialize_block_table(bytes, config)
        kwargs.update(
            {
                "stage_id": stage_id,
                "seqlens_q": seqlens_q,
                "seqlens_kv": seqlens_kv,
                "max_seqlen_q": max_seqlen_q,
                "max_seqlen_kv": max_seqlen_kv,
                "attn_mask": attn_mask,
                "q_block_table": q_block_table,
                "kv_block_table": kv_block_table,
                "out_block_table": out_block_table,
                "dq_block_table": dq_block_table,
                "dkv_block_table": dkv_block_table,
            }
        )
        return kwargs, bytes


@dataclass
class ReductionOp:
    src_buffers: List[Tuple[BufferBlock, BufferBlock]]
    dst_buffer: Tuple[BufferBlock, BufferBlock]

    def __post_init__(self):
        # check that out and lse buffer have the same index
        assert self.dst_buffer[0].index == self.dst_buffer[1].index
        for src_out, src_lse in self.src_buffers:
            if src_out.index != src_lse.index:
                raise ValueError(
                    f"src_out and src_lse buffer indices do not match: {src_out.index} != {src_lse.index}"
                )

    def serialize(self, config=SerializationConfig()) -> bytes:
        return (
            int_meta_to_bytes(len(self.src_buffers), config)
            + b"".join(
                [
                    serialize_buffer_block(src0, config)
                    + serialize_buffer_block(src1, config)
                    for src0, src1 in self.src_buffers
                ]
            )
            + serialize_buffer_block(self.dst_buffer[0], config)
            + serialize_buffer_block(self.dst_buffer[1], config)
        )

    @classmethod
    def _deserialize(
        cls, bytes, config=SerializationConfig()
    ) -> Tuple[Dict[str, Any], bytes]:
        n_src_buffers, bytes = bytes_to_int_meta(bytes, config)
        src_buffers = []
        for _ in range(n_src_buffers):
            src0, bytes = deserialize_buffer_block(bytes, config)
            src1, bytes = deserialize_buffer_block(bytes, config)
            src_buffers.append((src0, src1))
        dst0, bytes = deserialize_buffer_block(bytes, config)
        dst1, bytes = deserialize_buffer_block(bytes, config)
        reduction_op = ReductionOp(src_buffers, (dst0, dst1))
        return reduction_op, bytes


class AttnReductionInstr(BlockInstrBase):
    def __init__(
        self,
        ops: List[ReductionOp],
        output_is_unpadded: int,
        **kwargs,
    ):
        super().__init__(
            ops=ops,
            output_is_unpadded=output_is_unpadded,
            **kwargs,
        )
        self.ops: List[ReductionOp]
        self.output_is_unpadded: int

    def serialize(self, config=SerializationConfig()) -> bytes:
        return (
            super().serialize(config=config)
            + int_meta_to_bytes(len(self.ops), config)
            + b"".join([op.serialize(config) for op in self.ops])
            + int_meta_to_bytes(int(self.output_is_unpadded), config)
        )

    @classmethod
    def _deserialize(
        cls, bytes, config=SerializationConfig()
    ) -> Tuple[Dict[str, Any], bytes]:
        kwargs, bytes = super()._deserialize(bytes, config=config)
        n_ops, bytes = bytes_to_int_meta(bytes, config)
        ops = []
        for _ in range(n_ops):
            op, bytes = ReductionOp._deserialize(bytes, config=config)
            ops.append(op)
        output_is_unpadded, bytes = bytes_to_int_meta(bytes, config)
        kwargs.update(
            {
                "ops": ops,
                "output_is_unpadded": output_is_unpadded,
            }
        )
        return kwargs, bytes


@dataclass
class SumOp:
    src_buffers: List[BufferBlock]
    dst_buffer: BufferBlock

    def serialize(self, config=SerializationConfig()) -> bytes:
        return (
            int_meta_to_bytes(len(self.src_buffers), config)
            + b"".join(
                [
                    serialize_buffer_block(src, config)
                    for src in self.src_buffers
                ]
            )
            + serialize_buffer_block(self.dst_buffer, config)
        )

    @classmethod
    def _deserialize(
        cls, bytes, config=SerializationConfig()
    ) -> Tuple[Dict[str, Any], bytes]:
        n_src_buffers, bytes = bytes_to_int_meta(bytes, config)
        src_buffers = []
        for _ in range(n_src_buffers):
            src, bytes = deserialize_buffer_block(bytes, config)
            src_buffers.append(src)
        dst, bytes = deserialize_buffer_block(bytes, config)
        sum_op = SumOp(src_buffers, dst)
        return sum_op, bytes


class SumInstr(BlockInstrBase):
    def __init__(
        self,
        ops: List[SumOp],
        output_is_unpadded: int,
        **kwargs,
    ):
        super().__init__(
            ops=ops,
            output_is_unpadded=output_is_unpadded,
            **kwargs,
        )
        self.ops: List[SumOp]
        self.output_is_unpadded: int

    def serialize(self, config=SerializationConfig()) -> bytes:
        return (
            super().serialize(config=config)
            + int_meta_to_bytes(len(self.ops), config)
            + b"".join([op.serialize(config) for op in self.ops])
            + int_meta_to_bytes(int(self.output_is_unpadded), config)
        )

    @classmethod
    def _deserialize(
        cls, bytes, config=SerializationConfig()
    ) -> Tuple[Dict[str, Any], bytes]:
        kwargs, bytes = super()._deserialize(bytes, config=config)
        n_ops, bytes = bytes_to_int_meta(bytes, config)
        ops = []
        for _ in range(n_ops):
            op, bytes = SumOp._deserialize(bytes, config=config)
            ops.append(op)
        output_is_unpadded, bytes = bytes_to_int_meta(bytes, config)
        kwargs.update(
            {
                "ops": ops,
                "output_is_unpadded": output_is_unpadded,
            }
        )
        return kwargs, bytes


@dataclass
class BufferInfo:
    n_blocks: int
    block_size: int
    block_numel: int
    dtype: int
    buffer_shape: Tuple[int, ...]
    per_token_shape: Tuple[int, ...]


class ExecutionPlan:
    """
    Sequences of BlockInstructions to be executed on each device.
    """

    def __init__(
        self,
        instructions: List[BlockInstrBase],
        n_nodes: int,
        n_devices_per_node: int,
        node_id: int,
        local_device_id: int,
        n_stages: int,
        local_cu_seqlens: List[int],
        buffer_info: Dict[str, BufferInfo],
    ):
        self.instructions = instructions
        self.n_nodes = n_nodes
        self.n_devices_per_node = n_devices_per_node
        self.node_id = node_id
        self.local_device_id = local_device_id
        self.n_stages = n_stages
        self.local_cu_seqlens = local_cu_seqlens
        self.buffer_info = buffer_info

    def __repr__(self) -> str:
        return (
            "ExecutionPlan(n_nodes={}, n_devices_per_node={}, node_id={}, local_device_id={}, "
            "buffer_info={}, n_stages: {}, local_cu_seqlens: {}, instructions={})".format(
                self.n_nodes,
                self.n_devices_per_node,
                self.node_id,
                self.local_device_id,
                self.buffer_info,
                self.n_stages,
                self.local_cu_seqlens,
                self.instructions,
            )
        )

    def __str__(self):
        """Print the execution plan in a human readable format."""
        return (
            "ExecutionPlan(n_nodes={}, n_devices_per_node={}, node_id={}, local_device_id={}, "
            "buffer_info={}, n_stages: {}, local_cu_seqlens: {}, instructions=[\n\t".format(
                self.n_nodes,
                self.n_devices_per_node,
                self.node_id,
                self.local_device_id,
                self.buffer_info,
                self.n_stages,
                self.local_cu_seqlens,
            )
            + "\n\t".join([str(x) for x in self.instructions])
            + "\n])"
        )

    def __eq__(self, other: "ExecutionPlan" | Any):
        if not isinstance(other, ExecutionPlan):
            return False
        return (
            self.n_nodes == other.n_nodes
            and self.n_devices_per_node == other.n_devices_per_node
            and self.node_id == other.node_id
            and self.local_device_id == other.local_device_id
            and self.buffer_info == other.buffer_info
            and self.n_stages == other.n_stages
            and self.local_cu_seqlens == other.local_cu_seqlens
            and self.instructions == other.instructions
        )

    def serialize(self, config=SerializationConfig()) -> bytes:
        """Serialize the execution plan to a byte array."""

        def _serialize_plan_meta(x: int):
            return x.to_bytes(
                config.EXECUTION_PLAN_META_BYTES, config.BYTES_ENDIANNESS
            )

        return (
            _serialize_plan_meta(self.n_nodes)
            + _serialize_plan_meta(self.n_devices_per_node)
            + _serialize_plan_meta(self.node_id)
            + _serialize_plan_meta(self.local_device_id)
            + _serialize_plan_meta(self.n_stages)
            + serialize_list_of_ints(self.local_cu_seqlens, config)
            + _serialize_plan_meta(len(self.buffer_info))
            + b"".join(
                [
                    serialize_str(buffer_name, config)
                    + int_meta_to_bytes(buf_info.n_blocks, config)
                    + int_meta_to_bytes(buf_info.block_size, config)
                    + int_meta_to_bytes(buf_info.block_numel, config)
                    + int_meta_to_bytes(buf_info.dtype, config)
                    + serialize_int_tuple(buf_info.buffer_shape, config)
                    + serialize_int_tuple(buf_info.per_token_shape, config)
                    for buffer_name, buf_info in self.buffer_info.items()
                ]
            )
            + _serialize_plan_meta(len(self.instructions))
            + b"".join(
                [instr.serialize(config) for instr in self.instructions]
            )
        )

    @classmethod
    def deserialize(cls, bytes, config=SerializationConfig()) -> ExecutionPlan:
        """Deserialize the execution plan from a byte array."""

        def _deserialize_plan_meta(bytes):
            return (
                int.from_bytes(
                    bytes[: config.EXECUTION_PLAN_META_BYTES],
                    config.BYTES_ENDIANNESS,
                ),
                bytes[config.EXECUTION_PLAN_META_BYTES :],
            )

        n_nodes, bytes = _deserialize_plan_meta(bytes)
        n_devices_per_node, bytes = _deserialize_plan_meta(bytes)
        node_id, bytes = _deserialize_plan_meta(bytes)
        local_device_id, bytes = _deserialize_plan_meta(bytes)
        n_stages, bytes = _deserialize_plan_meta(bytes)
        local_cu_seqlens, bytes = deserialize_list_of_ints(bytes, config)
        n_buffers, bytes = _deserialize_plan_meta(bytes)
        buffer_info = {}
        for _ in range(n_buffers):
            buffer_name, bytes = deserialize_str(bytes, config)
            n_blocks, bytes = bytes_to_int_meta(bytes, config)
            block_size, bytes = bytes_to_int_meta(bytes, config)
            block_numel, bytes = bytes_to_int_meta(bytes, config)
            dtype, bytes = bytes_to_int_meta(bytes, config)
            buffer_shape, bytes = deserialize_int_tuple(bytes, config)
            per_token_shape, bytes = deserialize_int_tuple(bytes, config)
            buffer_info[buffer_name] = BufferInfo(
                n_blocks,
                block_size,
                block_numel,
                dtype,
                buffer_shape,
                per_token_shape,
            )
        n_instructions, bytes = _deserialize_plan_meta(bytes)
        instructions = []
        for _ in range(n_instructions):
            instr, bytes = BlockInstrBase.deserialize(bytes, config=config)
            instructions.append(instr)
        return (
            cls(
                instructions,
                n_nodes,
                n_devices_per_node,
                node_id,
                local_device_id,
                n_stages,
                local_cu_seqlens,
                buffer_info,
            ),
            bytes,
        )


def serialize_list_of_execution_plans(
    execution_plans: List[ExecutionPlan], config=SerializationConfig()
) -> bytes:
    return int_meta_to_bytes(len(execution_plans), config) + b"".join(
        [plan.serialize(config) for plan in execution_plans]
    )


def deserialize_list_of_execution_plans(
    bytes: bytes, config=SerializationConfig()
) -> Tuple[List[ExecutionPlan], bytes]:
    n_plans, bytes = bytes_to_int_meta(bytes, config)
    execution_plans = []
    for _ in range(n_plans):
        plan, bytes = ExecutionPlan.deserialize(bytes, config)
        execution_plans.append(plan)
    return execution_plans, bytes
