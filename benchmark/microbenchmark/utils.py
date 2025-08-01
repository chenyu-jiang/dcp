import os

import torch
import torch.distributed as dist
from torch.nn.utils.rnn import pad_sequence


# copied from https://github.com/InternLM/InternEvo.git
# internlm/model/ops/utils.py
def unpack_qkv_before_attn(
    cur_input: torch.Tensor, cu_seqlens: torch.Tensor, padding_v: int = 0
):
    """
    qkv: the shape is (1, packed_length, three, head_num, head_dim)
    kv: the shape is (1, packed_length, two, head_num, head_dim)
    q/k/v: the shape is (1, packed_length, head_num, head_dim)

    Return:
    output: the shape is (micro_bsz, seq_len, three, head_num, head_dim) for qkv
                        (micro_bsz, seq_len, two, head_num, head_dim) for kv
                        (micro_bsz, seq_len, head_num, head_dim) for q/k/v
    """
    assert cur_input.shape[0] == 1
    cur_input = cur_input.squeeze(0)

    sequences = []
    for i in range(len(cu_seqlens) - 1):
        sequences.append(cur_input[cu_seqlens[i] : cu_seqlens[i + 1]])

    padded_sequences = pad_sequence(
        sequences, batch_first=True, padding_value=padding_v
    )

    return padded_sequences


def print_all(s):
    for rank in range(dist.get_world_size()):
        if rank == dist.get_rank():
            print(f"RANK {rank}: {s}")
        dist.barrier(device_ids=[torch.cuda.current_device()])


def print_0(s):
    if dist.get_rank() == 0:
        print(s)


def init_env():
    local_rank = os.environ.get("LOCAL_RANK")
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    torch.set_default_device(device)
    print_all(f"Rank {dist.get_rank()} set device {device}")


def extract_local_zigzag(value, cu_seqlens, rank, world_size):
    local_values = []
    for i in range(len(cu_seqlens) - 1):
        start, end = cu_seqlens[i].item(), cu_seqlens[i + 1].item()
        local_value = value[start:end].chunk(2 * world_size, dim=0)
        local_values.extend(
            [
                local_value[rank].detach().clone(),
                local_value[2 * world_size - 1 - rank].detach().clone(),
            ]
        )
    return torch.cat(local_values, dim=0).contiguous()


def extract_local_zigzag_as_padded(value, cu_seqlens, rank, world_size):
    padded_value = unpack_qkv_before_attn(value.unsqueeze(0), cu_seqlens)
    padded_value_chunks = padded_value.chunk(2 * world_size, dim=1)
    return torch.cat(
        [
            padded_value_chunks[rank],
            padded_value_chunks[2 * world_size - 1 - rank],
        ],
        dim=1,
    )


def zero_out_local_padding_zigzag(
    local_padded, raw_global_seqlens, max_global_seqlen, rank, world_size
):
    chunk_seqlens = []
    chunk_size = max_global_seqlen // (2 * world_size)
    for seqlen in raw_global_seqlens:
        n_chunks = (seqlen + chunk_size - 1) // chunk_size
        curr_seq_seqlens = []
        if rank < n_chunks:
            curr_seq_seqlens.append(
                min(chunk_size, seqlen - rank * chunk_size)
            )
        else:
            curr_seq_seqlens.append(0)
        if 2 * world_size - 1 - rank < n_chunks:
            curr_seq_seqlens.append(
                min(
                    chunk_size,
                    seqlen - (2 * world_size - 1 - rank) * chunk_size,
                )
            )
        else:
            curr_seq_seqlens.append(0)
        chunk_seqlens.append(curr_seq_seqlens)
    local_chunks = local_padded.clone().chunk(2, dim=1)
    for b, (chunk1len, chunk2len) in enumerate(chunk_seqlens):
        local_chunks[0][b, chunk1len:].zero_()
        local_chunks[1][b, chunk2len:].zero_()
    return torch.cat(local_chunks, dim=1)


def extract_local_ring(value, cu_seqlens, rank, world_size):
    local_values = []
    for i in range(len(cu_seqlens) - 1):
        start, end = cu_seqlens[i].item(), cu_seqlens[i + 1].item()
        local_value = (
            value[start:end].chunk(world_size, dim=0)[rank].detach().clone()
        )
        local_values.append(local_value)
    return torch.cat(local_values, dim=0).contiguous()


def get_lse_diff(flash_attn_lse, cp_attn_lse, cu_seqlens, cp_type, cp_group):
    if cp_type == "zigzag":
        extract_out_func = extract_local_zigzag
    elif cp_type == "ring":
        extract_out_func = extract_local_ring
    else:
        raise ValueError(f"Unknown cp_type {cp_type}")
    flash_attn_lse_local = extract_out_func(
        flash_attn_lse.transpose(-2, -1),
        cu_seqlens,
        dist.get_rank(group=cp_group),
        dist.get_world_size(group=cp_group),
    ).transpose(-2, -1)
    lse_diff = (flash_attn_lse_local - cp_attn_lse).abs()
    lse_diff_mean = lse_diff.mean().item()
    lse_diff_max = lse_diff.max().item()
    return lse_diff_mean, lse_diff_max


def benchmark_f(
    f,
    warmup,
    iters,
    profile=False,
    profiler="nsys",
    profiler_out_dir=None,
    use_cudagraph=False,
):
    if profile:
        iters = 20
    if use_cudagraph:
        # first run once without cudagraph for some compilation to happen
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            f()
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        # capture
        with torch.cuda.graph(g):
            f()
        run_fn = g.replay
    else:
        run_fn = f
    for _ in range(warmup):
        run_fn()
    torch.cuda.synchronize()
    torch_profiler = None
    if profile:
        # barrier before starting profiling
        dist.barrier(device_ids=[torch.cuda.current_device()])
        if profiler == "nsys":
            torch.cuda.profiler.cudart().cudaProfilerStart()
        else:
            assert (
                profiler_out_dir is not None
            ), "profiler_out_dir must be set for torch profiler"
            torch_profiler = torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CUDA,
                    torch.profiler.ProfilerActivity.CPU,
                ],
                record_shapes=True,
                with_stack=True,
            )
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    if profile and torch_profiler:
        torch_profiler.start()
    start.record()
    for i in range(iters):
        if profile:
            torch.cuda.nvtx.range_push(f"iteration {i}")
        run_fn()
        if profile:
            torch.cuda.nvtx.range_pop()
    end.record()
    torch.cuda.synchronize()
    if profile:
        if torch_profiler:
            torch_profiler.stop()
            torch_profiler.export_chrome_trace(
                os.path.join(profiler_out_dir, f"trace_{dist.get_rank()}.json")
            )
            print_all(
                f"Profiler output saved to {profiler_out_dir}/trace_{dist.get_rank()}.json"
            )
        dist.barrier(device_ids=[torch.cuda.current_device()])
        torch.cuda.profiler.cudart().cudaProfilerStop()
    time = start.elapsed_time(end) / iters
    all_device_times = [
        torch.zeros(1, dtype=torch.float32)
        for _ in range(dist.get_world_size())
    ]
    dist.all_gather(
        all_device_times, torch.tensor([time], dtype=torch.float32)
    )
    all_device_times = torch.cat(all_device_times)
    avg_time = all_device_times.mean().item()
    max_time = all_device_times.max().item()
    return avg_time, max_time
