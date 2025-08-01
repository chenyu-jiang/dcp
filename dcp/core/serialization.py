import struct
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch


@dataclass
class SerializationConfig:
    BYTES_ENDIANNESS = "little"
    FLOAT_BYTES_ENDIANNESS = "<"
    # length of the serialized fields in bytes
    # increase these number if more bits are needed
    INSTRUCTION_INDEX_BYTES = 1
    EXECUTION_PLAN_META_BYTES = 4


def int_meta_to_bytes(n: int, config=SerializationConfig()) -> bytes:
    """Convert an integer to a byte array."""
    return n.to_bytes(
        config.EXECUTION_PLAN_META_BYTES, config.BYTES_ENDIANNESS
    )


def bytes_to_int_meta(
    bytes: bytes, config=SerializationConfig()
) -> Tuple[int, bytes]:
    """Convert a byte array to an integer."""
    return (
        int.from_bytes(
            bytes[: config.EXECUTION_PLAN_META_BYTES], config.BYTES_ENDIANNESS
        ),
        bytes[config.EXECUTION_PLAN_META_BYTES :],
    )


def serialize_list_of_ints(
    ints: List[int], config=SerializationConfig()
) -> bytes:
    """Serialize a list of ints to a byte array."""
    res = b""
    res += int_meta_to_bytes(len(ints), config)
    for i in ints:
        res += int_meta_to_bytes(i, config)
    return res


def deserialize_list_of_ints(
    bytes, config=SerializationConfig()
) -> Tuple[List[int], bytes]:
    """Deserialize a list of ints from a byte array."""
    n_ints, bytes = bytes_to_int_meta(bytes, config)
    ints = []
    for _ in range(n_ints):
        i, bytes = bytes_to_int_meta(bytes, config)
        ints.append(i)
    return ints, bytes


def serialize_str(s: str, config=SerializationConfig()) -> bytes:
    """Serialize a string to a byte array."""
    if not s:
        return int_meta_to_bytes(0, config)
    return (
        len(s.encode()).to_bytes(
            config.EXECUTION_PLAN_META_BYTES, config.BYTES_ENDIANNESS
        )
        + s.encode()
    )


def deserialize_str(
    bytes: bytes, config=SerializationConfig()
) -> Tuple[str, bytes]:
    """Deserialize a string from a byte array."""
    str_len, bytes = bytes_to_int_meta(bytes, config)
    if str_len == 0:
        return "", bytes
    s = bytes[:str_len].decode()
    return s, bytes[str_len:]


def serialize_int_tuple(
    t: Tuple[int, ...], config=SerializationConfig()
) -> bytes:
    """Serialize a tuple of ints to a byte array."""
    return len(t).to_bytes(
        config.EXECUTION_PLAN_META_BYTES, config.BYTES_ENDIANNESS
    ) + b"".join([int_meta_to_bytes(i, config) for i in t])


def deserialize_int_tuple(
    bytes: bytes, config=SerializationConfig()
) -> Tuple[Tuple[int, ...], bytes]:
    """Deserialize a tuple of ints from a byte array."""
    n_ints, bytes = bytes_to_int_meta(bytes, config)
    ints = []
    for _ in range(n_ints):
        i, bytes = bytes_to_int_meta(bytes, config)
        ints.append(i)
    return tuple(ints), bytes


def serialize_float(f: float, config=SerializationConfig()) -> bytes:
    """Serialize a float to a byte array."""
    # Pack the float as a double precision (8 bytes) using the specified endianness
    return struct.pack(config.FLOAT_BYTES_ENDIANNESS + "d", f)


def deserialize_float(
    byte_data: bytes, config=SerializationConfig()
) -> Tuple[float, bytes]:
    """Deserialize a float from a byte array."""
    # Unpack the first 8 bytes as a double precision float
    float_value = struct.unpack(
        config.FLOAT_BYTES_ENDIANNESS + "d", byte_data[:8]
    )[0]
    # Return the float and the remaining bytes
    return float_value, byte_data[8:]


def serialize_list_of_floats(
    floats: List[float], config=SerializationConfig()
) -> bytes:
    """Serialize a list of floats to a byte array."""
    res = b""
    res += int_meta_to_bytes(len(floats), config)
    for f in floats:
        res += serialize_float(f, config)
    return res


def deserialize_list_of_floats(
    bytes: bytes, config=SerializationConfig()
) -> Tuple[List[float], bytes]:
    """Deserialize a list of floats from a byte array."""
    n_floats, bytes = bytes_to_int_meta(bytes, config)
    floats = []
    for _ in range(n_floats):
        f, bytes = deserialize_float(bytes, config)
        floats.append(f)
    return floats, bytes


def serialize_list_of_list_of_list_of_ints(
    ints: List[List[List[int]]], config=SerializationConfig()
) -> bytes:
    """Serialize a list of lists of lists of ints to a byte array."""
    res = b""
    res += int_meta_to_bytes(len(ints), config)
    for inner_list in ints:
        res += serialize_list_of_list_of_ints(inner_list, config)
    return res


def deserialize_list_of_list_of_list_of_ints(
    bytes: bytes, config=SerializationConfig()
) -> Tuple[List[List[List[int]]], bytes]:
    """Deserialize a list of lists of lists of ints from a byte array."""
    n_lists, bytes = bytes_to_int_meta(bytes, config)
    lists = []
    for _ in range(n_lists):
        inner_list, bytes = deserialize_list_of_list_of_ints(bytes, config)
        lists.append(inner_list)
    return lists, bytes


def serialize_dict_of_tuple_int_to_dict_of_int_to_int(
    d: Dict[Tuple[int, int], Dict[int, int]], config=SerializationConfig()
) -> bytes:
    """Serialize a dictionary of tuples of ints to a dictionary of ints to ints."""
    res = b""
    res += int_meta_to_bytes(len(d), config)
    for (k1, k2), v in d.items():
        res += int_meta_to_bytes(k1, config)
        res += int_meta_to_bytes(k2, config)
        res += int_meta_to_bytes(len(v), config)
        for k3, v2 in v.items():
            res += int_meta_to_bytes(k3, config)
            res += int_meta_to_bytes(v2, config)

    return res


def deserialize_dict_of_tuple_int_to_dict_of_int_to_int(
    bytes: bytes, config=SerializationConfig()
) -> Tuple[Dict[Tuple[int, int], Dict[int, int]], bytes]:
    """Deserialize a dictionary of tuples of ints to a dictionary of ints to ints."""
    d = {}
    n_items, bytes = bytes_to_int_meta(bytes, config)
    for _ in range(n_items):
        k1, bytes = bytes_to_int_meta(bytes, config)
        k2, bytes = bytes_to_int_meta(bytes, config)
        v = {}
        n_inner_items, bytes = bytes_to_int_meta(bytes, config)
        for _ in range(n_inner_items):
            k3, bytes = bytes_to_int_meta(bytes, config)
            v2, bytes = bytes_to_int_meta(bytes, config)
            v[k3] = v2
        d[(k1, k2)] = v
    return d, bytes


def serialize_dict_of_tuple_int_to_list_of_ints(
    d: Dict[Tuple[int, int], List[int]], config=SerializationConfig()
) -> bytes:
    """Serialize a dictionary of tuples of ints to a list of ints."""
    res = b""
    res += int_meta_to_bytes(len(d), config)
    for (k1, k2), v in d.items():
        res += int_meta_to_bytes(k1, config)
        res += int_meta_to_bytes(k2, config)
        res += serialize_list_of_ints(v, config)
    return res


def deserialize_dict_of_tuple_int_to_list_of_ints(
    bytes: bytes, config=SerializationConfig()
) -> Tuple[Dict[Tuple[int, int], List[int]], bytes]:
    """Deserialize a dictionary of tuples of ints to a list of ints."""
    d = {}
    n_items, bytes = bytes_to_int_meta(bytes, config)
    for _ in range(n_items):
        k1, bytes = bytes_to_int_meta(bytes, config)
        k2, bytes = bytes_to_int_meta(bytes, config)
        v, bytes = deserialize_list_of_ints(bytes, config)
        d[(k1, k2)] = v
    return d, bytes


def serialize_dict_of_int_to_int(
    d: Dict[int, int], config=SerializationConfig()
) -> bytes:
    """Serialize a dictionary of ints to infinity."""
    res = b""
    res += int_meta_to_bytes(len(d), config)
    for k, v in d.items():
        res += int_meta_to_bytes(k, config)
        res += int_meta_to_bytes(v, config)
    return res


def deserialize_dict_of_int_to_int(
    bytes: bytes, config=SerializationConfig()
) -> Tuple[Dict[int, int], bytes]:
    d = {}
    n_items, bytes = bytes_to_int_meta(bytes, config)
    for _ in range(n_items):
        k, bytes = bytes_to_int_meta(bytes, config)
        v, bytes = bytes_to_int_meta(bytes, config)
        d[k] = v
    return d, bytes


def serialize_dict_of_str_to_str(
    d: Dict[str, str], config=SerializationConfig()
) -> bytes:
    """Serialize a dictionary of strings to a dictionary of strings."""
    res = b""
    res += int_meta_to_bytes(len(d), config)
    for k, v in d.items():
        res += serialize_str(k, config)
        res += serialize_str(v, config)
    return res


def deserialize_dict_of_str_to_str(
    bytes: bytes, config=SerializationConfig()
) -> Tuple[Dict[str, str], bytes]:
    """Deserialize a dictionary of strings to a dictionary of strings."""
    d = {}
    n_items, bytes = bytes_to_int_meta(bytes, config)
    for _ in range(n_items):
        k, bytes = deserialize_str(bytes, config)
        v, bytes = deserialize_str(bytes, config)
        d[k] = v
    return d, bytes


def serialize_dict_of_str_to_int(
    d: Dict[str, int], config=SerializationConfig()
) -> bytes:
    """Serialize a dictionary of strings to a dictionary of ints."""
    res = b""
    res += int_meta_to_bytes(len(d), config)
    for k, v in d.items():
        res += serialize_str(k, config)
        res += int_meta_to_bytes(v, config)
    return res


def deserialize_dict_of_str_to_int(
    bytes: bytes, config=SerializationConfig()
) -> Tuple[Dict[str, int], bytes]:
    """Deserialize a dictionary of strings to a dictionary of ints."""
    d = {}
    n_items, bytes = bytes_to_int_meta(bytes, config)
    for _ in range(n_items):
        k, bytes = deserialize_str(bytes, config)
        v, bytes = bytes_to_int_meta(bytes, config)
        d[k] = v
    return d, bytes


def serialize_tensor(t: torch.Tensor, config=SerializationConfig()) -> bytes:
    """Serialize a PyTorch tensor to a byte array."""
    # Serialize the tensor's shape
    t_bytes = t.cpu().numpy().tobytes()
    bytes = b"".join(
        [
            int_meta_to_bytes(torch_dtype_to_int(t.dtype), config),
            int_meta_to_bytes(len(t_bytes), config),
            t_bytes,
        ]
    )

    return bytes


def deserialize_tensor(
    bytes: bytes, config=SerializationConfig()
) -> Tuple[torch.Tensor, bytes]:
    """Deserialize a PyTorch tensor from a byte array."""
    # Deserialize the tensor's shape
    dtype, bytes = bytes_to_int_meta(bytes, config)
    dtype = int_to_torch_dtype(dtype)
    t_len, bytes = bytes_to_int_meta(bytes, config)
    t_bytes = bytes[:t_len]
    bytes = bytes[t_len:]
    t = torch.frombuffer(bytearray(t_bytes), dtype=dtype)

    return t, bytes


def serialize_list_of_list_of_ints(
    ints: List[List[int]], config=SerializationConfig()
) -> bytes:
    """Serialize a list of lists of ints to a byte array."""
    res = b""
    res += int_meta_to_bytes(len(ints), config)
    for inner_list in ints:
        res += serialize_list_of_ints(inner_list, config)
    return res


def deserialize_list_of_list_of_ints(
    bytes: bytes, config=SerializationConfig()
) -> Tuple[List[List[int]], bytes]:
    """Deserialize a list of lists of ints from a byte array."""
    n_lists, bytes = bytes_to_int_meta(bytes, config)
    lists = []
    for _ in range(n_lists):
        inner_list, bytes = deserialize_list_of_ints(bytes, config)
        lists.append(inner_list)
    return lists, bytes


def deserialize_list_of_list_of_ints(
    bytes: bytes, config=SerializationConfig()
) -> Tuple[List[List[int]], bytes]:
    """Deserialize a list of lists of ints from a byte array."""
    n_lists, bytes = bytes_to_int_meta(bytes, config)
    lists = []
    for _ in range(n_lists):
        inner_list, bytes = deserialize_list_of_ints(bytes, config)
        lists.append(inner_list)
    return lists, bytes


def serialize_dict_of_int_to_list_of_ints(
    d: Dict[int, List[int]], config=SerializationConfig()
) -> bytes:
    """Serialize a dictionary of ints to a list of ints."""
    res = b""
    res += int_meta_to_bytes(len(d), config)
    for k, v in d.items():
        res += int_meta_to_bytes(k, config)
        res += serialize_list_of_ints(v, config)
    return res


def deserialize_dict_of_int_to_list_of_ints(
    bytes: bytes, config=SerializationConfig()
) -> Tuple[Dict[int, List[int]], bytes]:
    """Deserialize a dictionary of ints to a list of ints."""
    d = {}
    n_items, bytes = bytes_to_int_meta(bytes, config)
    for _ in range(n_items):
        k, bytes = bytes_to_int_meta(bytes, config)
        v, bytes = deserialize_list_of_ints(bytes, config)
        d[k] = v
    return (d,)


def deserialize_dict_of_int_to_list_of_tuples(
    bytes: bytes, config=SerializationConfig()
) -> Tuple[Dict[int, List[Tuple[int, int]]], bytes]:
    """Deserialize a dictionary of ints to a list of tuples of ints."""
    d = {}
    n_items, bytes = bytes_to_int_meta(bytes, config)
    for _ in range(n_items):
        k, bytes = bytes_to_int_meta(bytes, config)
        v, bytes = deserialize_list_of_ints(bytes, config)
        v = tuple(v)
        d[k] = v
    return d, bytes


def torch_dtype_to_int(dtype: torch.dtype) -> int:
    if dtype == torch.bfloat16:
        return 0
    elif dtype == torch.float16:
        return 1
    elif dtype == torch.float32:
        return 2
    elif dtype == torch.int32:
        return 3
    elif dtype == torch.int64:
        return 4
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")


def int_to_torch_dtype(dtype: int) -> torch.dtype:
    if dtype == 0:
        return torch.bfloat16
    elif dtype == 1:
        return torch.float16
    elif dtype == 2:
        return torch.float32
    elif dtype == 3:
        return torch.int32
    elif dtype == 4:
        return torch.int64
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")
