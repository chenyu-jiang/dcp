import pickle
from typing import List

import numpy as np
import torch
import torch.utils.benchmark as benchmark

from dcp.ops.fused_ops import _fused_copy_bf16


def benchmark_forward(
    fn,
    *inputs,
    repeats=10,
    desc="",
    verbose=True,
    amp=False,
    amp_dtype=torch.float16,
    **kwinputs,
):
    """Use Pytorch Benchmark on the forward pass of an arbitrary function."""
    if verbose:
        print(desc, "- Forward pass")

    def amp_wrapper(*inputs, **kwinputs):
        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp):
            fn(*inputs, **kwinputs)

    t = benchmark.Timer(
        stmt="fn_amp(*inputs, **kwinputs)",
        globals={
            "fn_amp": amp_wrapper,
            "inputs": inputs,
            "kwinputs": kwinputs,
        },
        num_threads=torch.get_num_threads(),
    )
    m = t.timeit(repeats)
    if verbose:
        print(m)
    return t, m


def generate_blocks(n_blocks_per_size: List[int], block_sizes: List[int]):
    buffers = []
    dst_buffers = []

    src_ptrs = []
    dst_ptrs = []
    src_offsets = []
    dst_offsets = []
    src_strides = []
    dst_strides = []
    for block_size, n_blocks in zip(block_sizes, n_blocks_per_size):
        buffer = torch.randn(n_blocks * 2, block_size)
        buffers.append(buffer)
        dst_buffer = torch.zeros_like(buffer)
        dst_buffers.append(dst_buffer)
        buffer_ptrs = torch.tensor(
            [buffer.data_ptr() for _ in range(n_blocks)]
        )
        dst_buffer_ptrs = torch.tensor(
            [dst_buffer.data_ptr() for _ in range(n_blocks)]
        )
        src_ptrs.append(buffer_ptrs)
        dst_ptrs.append(dst_buffer_ptrs)
        block_offsets = torch.tensor(
            np.random.choice(n_blocks * 2, n_blocks, replace=False),
            dtype=torch.int32,
        )
        dst_block_offsets = torch.tensor(
            np.random.choice(n_blocks * 2, n_blocks, replace=False),
            dtype=torch.int32,
        )
        block_strides = torch.tensor(
            [block_size for _ in range(n_blocks)], dtype=torch.int32
        )
        dst_block_strides = torch.tensor(
            [block_size for _ in range(n_blocks)], dtype=torch.int32
        )
        src_offsets.append(block_offsets)
        dst_offsets.append(dst_block_offsets)
        src_strides.append(block_strides)
        dst_strides.append(dst_block_strides)
    return (
        src_ptrs,
        src_offsets,
        src_strides,
        buffers,
        dst_ptrs,
        dst_offsets,
        dst_strides,
        dst_buffers,
    )


@torch.compile
def ref_copy(
    src_buffers: List[torch.Tensor],
    src_offsets: List[torch.Tensor],
    dst_buffers: List[torch.Tensor],
    dst_offsets: List[torch.Tensor],
):
    for src_buffer, src_block_offsets, dst_buffer, dst_block_offsets in zip(
        src_buffers, src_offsets, dst_buffers, dst_offsets
    ):
        dst_buffer[dst_block_offsets] = src_buffer[src_block_offsets]


def test_fused_copy(
    n_sizes: int, n_blocks_per_size: int, triton_block_size: int
):
    block_size: int = 256 * 128
    torch.set_default_dtype(torch.bfloat16)
    torch.set_default_device("cuda")
    block_sizes = [block_size] * n_sizes
    n_blocks_per_size_list = [n_blocks_per_size] * n_sizes
    (
        src_ptrs,
        src_offsets,
        src_strides,
        buffers,
        dst_ptrs,
        dst_offsets,
        dst_strides,
        dst_buffers,
    ) = generate_blocks(n_blocks_per_size_list, block_sizes)
    ref_dst_buffers = [buffer.clone() for buffer in dst_buffers]

    def _run_ref_copy():
        ref_copy(buffers, src_offsets, ref_dst_buffers, dst_offsets)

    _run_ref_copy()
    ref_copy_g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(ref_copy_g):
        _run_ref_copy()

    def _run_triton_copy():
        for (
            block_size,
            src_buffer_ptrs,
            src_block_offsets,
            src_block_strides,
            dst_buffer_ptrs,
            dst_block_offsets,
            dst_block_strides,
        ) in zip(
            block_sizes,
            src_ptrs,
            src_offsets,
            src_strides,
            dst_ptrs,
            dst_offsets,
            dst_strides,
        ):
            n_programs_per_block = block_size // triton_block_size
            _fused_copy_bf16[(len(src_buffer_ptrs) * n_programs_per_block,)](
                src_buffer_ptrs,
                src_block_offsets,
                src_block_strides,
                dst_buffer_ptrs,
                dst_block_offsets,
                dst_block_strides,
                block_size,
                triton_block_size,
            )

    _run_triton_copy()
    triton_copy_g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(triton_copy_g):
        _run_triton_copy()

    torch_f = benchmark_forward(
        ref_copy_g.replay,
        repeats=30,
        verbose=False,
    )

    triton_f = benchmark_forward(
        triton_copy_g.replay,
        repeats=30,
        verbose=False,
    )
    return torch_f, triton_f


results = {}
for n_sizes in [1, 2, 4, 8]:
    for n_blocks_per_size in [2, 4, 8, 16, 32]:
        for triton_block_size in [64, 128, 256]:
            config = (n_sizes, n_blocks_per_size, triton_block_size)
            torch_f, triton_f = test_fused_copy(
                n_sizes, n_blocks_per_size, triton_block_size
            )
            print(
                f"===== n_sizes: {n_sizes}, n_blocks_per_size: {n_blocks_per_size}, triton_block_size: {triton_block_size}"
            )
            print("torch time:", torch_f[1].mean * 1e6)
            print("triton time:", triton_f[1].mean * 1e6)
            results[config, "Triton"] = triton_f[1].mean * 1e6
            results[config, "Pytorch"] = torch_f[1].mean * 1e6

with open("benchmark_fused_copy.pkl", "wb") as f:
    pickle.dump(results, f, pickle.HIGHEST_PROTOCOL)
