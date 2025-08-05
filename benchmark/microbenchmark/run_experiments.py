import argparse
import codecs
import json
import os
import re
import shutil
import subprocess
import time
from typing import IO

import torch
import torch.distributed as dist
import tqdm

from benchmark.utils import RedisKVStore, kill_redis_servers

FRAMEWORKS = ["ring", "zigzag", "te", "lt", "dcp"]
MASK_TYPES = ["causal", "lambda", "shared_question", "causal_blockwise"]
N_HEADS = 8
N_QUERY_GROUPS = 2
HEAD_DIM = 128


class NonBlockingReader:
    def __init__(self, f: IO[bytes], delimiters="\r\n"):
        self.delimiters = delimiters
        self.buffer = ""
        self.f_str = codecs.getreader("utf-8")(f)
        # Set the file descriptor to non-blocking mode
        os.set_blocking(f.fileno(), False)

    def readline(self):
        """
        Read a line from the file-like object with custom delimiters.
        """
        encountered_delimiter = False
        while True:
            try:
                c = self.f_str.read(1)
            except TypeError:
                # Handle the case where read() returns None
                c = None
            if c is None:
                break
            if c in self.delimiters:
                encountered_delimiter = True
                break
            self.buffer += c
        if encountered_delimiter:
            line = self.buffer
            self.buffer = ""
            return line
        return ""

    def extract_leftover(self):
        """
        Extract any leftover data in the buffer.
        """
        while True:
            try:
                c = self.f_str.read(1)
            except TypeError:
                # Handle the case where read() returns None
                c = None
            if c is None or c == "":
                break
            self.buffer += c
        # split buffer by delimiters
        delimiters_re = "|".join(
            re.escape(delimiter) for delimiter in self.delimiters
        )
        lines = re.split(delimiters_re, self.buffer)
        yield from lines


# naive way to test failure
def _test_failure(text: str):
    """
    Helper function to check if a exp has failed based on the output text.
    """
    keywords = ["failed", "broken pipe", "exception", "most recent call first"]
    for keyword in keywords:
        if keyword in text.lower():
            return True
    return False


def _report_failure(kv_store: RedisKVStore, node_rank: int):
    """
    Report failure to the kv store.
    """
    kv_store.set("failure", str(node_rank))


def _check_failure(kv_store: RedisKVStore):
    """
    Check if the kv store has reported a failure.
    """
    failure = kv_store.get("failure")
    if failure is not None and failure.decode("utf-8") != "-1":
        node_rank = int(failure.decode("utf-8"))
        return node_rank
    return None


def _reset_failure(kv_store: RedisKVStore):
    """
    Reset the failure flag in the kv store.
    """
    kv_store.set("failure", "-1")


def _is_progress(text: str):
    """
    Check if the text is a progress update.
    """
    # This is a very naive check, but it should work for most cases
    return (
        "Benchmark: " in text
        and "total time" not in text
        and "Benchmarking" not in text
    ) and not _test_failure(text)


def kill_benchmark_processes():
    """
    Kill all benchmark processes that are running.
    This is a very naive way to do it, but it should work for most cases.
    """
    subprocess.run(
        ["pkill", "-f", "benchmark/microbenchmark/benchmark_attention.py"]
    )


def run_cmd(
    kv_store: RedisKVStore,
    framework,
    benchmark_type,
    mask_type,
    seqlens_config,
    n_nodes,
    n_gpus_per_node,
    master_addr,
    master_port,
    node_rank,
    dcp_block_size,
    use_block_size_heuristic=False,
    lt_window_size=-1,
    te_cp_comm_type="a2a+p2p",
    a2a_degree=1,
    dp_degree=1,
    n_samples=500,
    log_dir="./logs",
    print_fn=print,
):
    current_file_dir = os.path.dirname(os.path.abspath(__file__))
    # create log directory if it does not exist
    exp_name = f"{framework}_{benchmark_type}_m{mask_type}"
    if framework == "lt":
        exp_name += f"_lt{lt_window_size}"
    if framework == "dcp":
        exp_name += f"_blk{dcp_block_size}"
    print_fn("Running experiment: " + exp_name)
    logging_subdir = os.path.join(log_dir, exp_name)
    if not os.path.exists(logging_subdir):
        os.makedirs(logging_subdir)
    log_file = os.path.join(logging_subdir, "stdout_stderr.log")
    if node_rank == 0:
        _reset_failure(kv_store)
    torch.distributed.barrier(device_ids=[torch.cuda.current_device()])
    s = (
        f"{current_file_dir}/run_distributed_benchmark.sh {n_nodes} {n_gpus_per_node} {master_addr} {master_port} {node_rank} "
        f"-b {framework} --seqlens-config {seqlens_config} --lt-window-size {lt_window_size} -bt {benchmark_type} "
        f"-t {te_cp_comm_type} --dp-degree {dp_degree} --block-size {dcp_block_size} --a2a-degree {a2a_degree} "
        f"--mask {mask_type} --n-heads {N_HEADS} --n-query-groups {N_QUERY_GROUPS} --head-dim {HEAD_DIM} --n-samples {n_samples}"
    )
    if use_block_size_heuristic:
        s += " --use-block-size-heuristic"
    p = subprocess.Popen(
        s, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
    )
    total_time = None
    reader = NonBlockingReader(p.stdout, delimiters="\r\n")
    with open(log_file, "a+") as log_f:
        while True:
            # test if p is still running
            retcode = p.poll()
            if retcode is not None:
                break
            nextline = reader.readline()
            if _is_progress(nextline):
                print_fn(nextline)
            # write the output to the log file
            if nextline != "":
                log_f.write(nextline + "\n")
                log_f.flush()
            if _test_failure(nextline):
                # this doesn't catch the case where the process is stuck
                # but hope it's good enough
                print_fn(f"Run failed on node {node_rank}: {nextline.strip()}")
                _report_failure(kv_store, node_rank)
                break
            if "total time" in nextline:
                splitted_line = nextline.strip().split()
                for i in range(len(splitted_line)):
                    if splitted_line[i] == "ms":
                        total_time = float(splitted_line[i - 1])
                        break
                break
            failure_status = _check_failure(kv_store)
            if failure_status is not None:
                print_fn(
                    f"Run failed due to failure on node {failure_status}."
                )
                break
            time.sleep(0.1)
        # wait 5 seconds and kill the process
        time.sleep(5)
        p.terminate()
        kill_benchmark_processes()
        kill_redis_servers(node_rank, kv_store)
        # if theres anything left in the buffer, read it
        for line in reader.extract_leftover():
            if line.strip() != "":
                if _is_progress(line):
                    print_fn(line.strip())
                log_f.write(line + "\n")
                log_f.flush()
    # barrier
    print_fn(f"Run finished on node {node_rank}.")
    dist.barrier(device_ids=[torch.cuda.current_device()])
    if framework == "dcp":
        # move the debug dir into a separate folder
        if os.path.exists("./debug"):
            dst_path = os.path.join(logging_subdir, f"dcp_debug")
            if os.path.exists(dst_path):
                # remove the old debug folder
                shutil.rmtree(dst_path)
            shutil.move(
                "./debug",
                os.path.join(
                    logging_subdir,
                    f"dcp_debug",
                ),
            )
    return total_time


def get_finished_exps(out_path):
    configs = set()
    with open(out_path, "r") as f:
        _ = f.readline()
        for line in f:
            (
                n_heads,
                n_query_groups,
                head_dim,
                framework,
                benchmark_type,
                mask_type,
                lt_window_size,
                a2a_degree,
                dcp_block_size,
                dp_degree,
                avg_time,
                n_iters,
            ) = line.strip().split(",")
            if avg_time != "None":
                configs.add(
                    (
                        int(n_heads),
                        int(n_query_groups),
                        int(head_dim),
                        framework,
                        benchmark_type,
                        mask_type,
                        int(lt_window_size),
                        int(a2a_degree),
                        int(dcp_block_size),
                        int(dp_degree),
                    )
                )
    return configs


def run_exp(
    kv_store: RedisKVStore,
    config_path,
    out_path,
    n_nodes,
    n_gpus_per_node,
    max_dp_degree,
    master_addr,
    master_port,
    node_rank,
    n_samples,
    log_dir="./logs",
):
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    configs_to_skip = set()
    if node_rank == 0:
        if os.path.exists(out_path):
            configs_to_skip = get_finished_exps(out_path)
            f = open(out_path, "a")
        else:
            f = open(out_path, "w")
            f.write(
                "n_heads,n_query_groups,head_dim,framework,benchmark_type,mask_type,lt_window_size,a2a_degree,dcp_block_size,dp_degree,avg_time,n_iters\n"
            )
    for framework in tqdm.tqdm(FRAMEWORKS, desc="Frameworks"):
        dp_degree = max_dp_degree if framework != "dcp" else 1
        if framework in ["dcp", "te"]:
            mask_types = MASK_TYPES
        else:
            mask_types = ["causal"]
        if framework == "lt":
            cp_degree = n_gpus_per_node * n_nodes // dp_degree
            head_degree = min(N_QUERY_GROUPS, cp_degree)
            ring_degree = n_nodes * n_gpus_per_node // head_degree // dp_degree
            lt_window_sizes = []
            window_size = 1
            while window_size <= ring_degree:
                lt_window_sizes.append(window_size)
                window_size *= 2
        else:
            lt_window_sizes = [-1]
        if framework == "te" or framework == "lt":
            cp_degree = n_gpus_per_node * n_nodes // dp_degree
            head_degree = min(N_QUERY_GROUPS, cp_degree)
            a2a_degrees = [head_degree]
        else:
            a2a_degrees = [1]
        if framework == "dcp":
            dcp_block_sizes = [512, 1024, 2048, 4096]
        else:
            dcp_block_sizes = [1]
        for benchmark_type in ["forward", "backward"]:
            for mask_type in mask_types:
                for lt_window_size in tqdm.tqdm(
                    lt_window_sizes, desc="LT Window Sizes", leave=False
                ):
                    for a2a_degree in tqdm.tqdm(
                        a2a_degrees, desc="A2A Degrees", leave=False
                    ):
                        for dcp_block_size in tqdm.tqdm(
                            dcp_block_sizes,
                            desc="DCP Block Sizes",
                            leave=False,
                        ):
                            if (
                                N_HEADS,
                                N_QUERY_GROUPS,
                                HEAD_DIM,
                                framework,
                                benchmark_type,
                                mask_type,
                                lt_window_size,
                                a2a_degree,
                                dcp_block_size,
                                dp_degree,
                            ) in configs_to_skip:
                                should_skip = True
                            else:
                                should_skip = False
                            obj_list = [should_skip]
                            dist.broadcast_object_list(
                                obj_list, src=0, device=torch.device("cuda")
                            )
                            should_skip = obj_list[0]
                            if should_skip:
                                continue
                            dist.barrier(
                                device_ids=[torch.cuda.current_device()]
                            )
                            avg_time = run_cmd(
                                kv_store,
                                framework,
                                benchmark_type,
                                mask_type,
                                config_path,
                                n_nodes,
                                n_gpus_per_node,
                                master_addr,
                                master_port,
                                node_rank,
                                dcp_block_size,
                                lt_window_size=lt_window_size,
                                a2a_degree=a2a_degree,
                                dp_degree=dp_degree,
                                n_samples=n_samples,
                                log_dir=log_dir,
                                print_fn=tqdm.tqdm.write,
                            )
                            dist.barrier(
                                device_ids=[torch.cuda.current_device()]
                            )
                            if node_rank == 0:
                                f.write(
                                    f"{N_HEADS},{N_QUERY_GROUPS},{HEAD_DIM},{framework},{benchmark_type},{mask_type},{lt_window_size},{a2a_degree},{dcp_block_size},{dp_degree},{avg_time},{n_samples}\n"
                                )
                                f.flush()
                            time.sleep(10)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("seqlens_paths", type=str, nargs="+")
    parser.add_argument("-n", "--n_nodes", type=int, default=1)
    parser.add_argument("-ng", "--n_gpus_per_node", type=int, default=4)
    parser.add_argument("-m", "--master_addr", type=str, required=True)
    parser.add_argument("-p", "--master_port", type=str, required=True)
    parser.add_argument("-nr", "--node_rank", type=int, default=0)
    parser.add_argument("-ns", "--n_samples", type=int, default=50)
    parser.add_argument("--out-dir", type=str, default="./microbenchmarks")
    args = parser.parse_args()

    kv_store = RedisKVStore(
        args.master_addr, args.node_rank, args.n_nodes, "microbenchmark"
    )

    for seqlens_path in args.seqlens_paths:
        seqlens_name = os.path.basename(seqlens_path).rsplit(".", 1)[0]
        exp_name = f"{seqlens_name}_{args.n_nodes}N{args.n_nodes * args.n_gpus_per_node}D"
        exp_dir = os.path.join(args.out_dir, exp_name)
        if not os.path.exists(exp_dir):
            os.makedirs(exp_dir)
        log_dir = os.path.join(exp_dir, "logs")
        output_path = os.path.join(exp_dir, "results.csv")

        with open(seqlens_path) as f:
            config = json.load(f)

        max_dp_degree = config["max_dp_degree"]
        # round down to the nearest power of 2
        dp_degree = 1
        while dp_degree * 2 <= max_dp_degree:
            dp_degree *= 2

        run_exp(
            kv_store,
            seqlens_path,
            output_path,
            args.n_nodes,
            args.n_gpus_per_node,
            dp_degree,
            args.master_addr,
            args.master_port,
            args.node_rank,
            args.n_samples,
            log_dir=log_dir,
        )
