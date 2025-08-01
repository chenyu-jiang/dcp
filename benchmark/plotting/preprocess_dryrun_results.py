import os
import argparse
from typing import List
from collections import defaultdict

from tqdm import tqdm
import pandas as pd

# assume use 8 KV heads, head dim = 128, TP size = 4, 8 GPUs per node
# so each TP rank initially has 2 KV heads
# 2 CP ranks on each node (forming head parallel)
# ring comm between 8 nodes
PER_TOKEN_SIZE_KV_ON_EACH_NODE = 2 * 2 * 128
# assume bf16
PER_TOKEN_BYTES_KV_ON_EACH_NODE = PER_TOKEN_SIZE_KV_ON_EACH_NODE * 2
RING_SIZE = 8
# there are RING_SIZE - 1 rounds
# in each round, each node sends PER_TOKEN_BYTES_KV_ON_EACH_NODE local KV to the next node
PER_TOKEN_COMM_PER_NODE_EACH_ROUND = PER_TOKEN_BYTES_KV_ON_EACH_NODE
# we count total comm volume within the entire cluster
TOTAL_COMM_PER_TOKEN = (
    PER_TOKEN_COMM_PER_NODE_EACH_ROUND * (RING_SIZE - 1) * RING_SIZE
)


def parse_results(exp_dir: str):
    if os.path.isdir(os.path.join(exp_dir, "compiler")):
        debug_dir = os.path.join(exp_dir, "compiler")
    else:
        assert os.path.isdir(
            exp_dir
        ), f"Experiment directory {exp_dir} does not exist."
        debug_dir = exp_dir

    logs = os.listdir(debug_dir)
    logs = [log for log in logs if log.endswith(".log")]
    logs.sort()
    block_size_all = None
    inter_node_costs = []
    max_memory_per_devices = []
    compute_imbalances = []
    compilation_times = []
    for log in logs:
        try:
            (
                block_size,
                inter_node_cost,
                max_memory_per_device,
                compute_imbalance,
                compilation_time,
            ) = parse_log(os.path.join(debug_dir, log))
        except Exception as e:
            iter_id = int(log.split(".")[0].split("_")[-1][4:])
            if iter_id < 200:
                print(f"Error parsing log {log}: {e}")
            continue
        if block_size_all is None:
            block_size_all = block_size
        assert block_size == block_size_all
        inter_node_costs.append(inter_node_cost)
        max_memory_per_devices.append(max_memory_per_device)
        compute_imbalances.append(compute_imbalance)
        compilation_times.append(compilation_time)
    res = (
        block_size_all,
        sum(inter_node_costs) / len(inter_node_costs),
        max(max_memory_per_devices),
        sum(compute_imbalances) / len(compute_imbalances),
        sum(compilation_times) / len(compilation_times),
    )
    return res


def _parse_n_tokens(log: str):
    try:
        parse_log(log)
    except Exception as e:
        # we skip incomplete logs
        raise e
    with open(log, "r") as f:
        lines = f.readlines()
    lines = [line.strip() for line in lines]
    lines = [line for line in lines if line]
    for line in lines:
        if "raw_seqlens:" in line:
            seqlens = line.split("raw_seqlens:")[1].strip()[1:-1].split(",")
            seqlens = [x.strip() for x in seqlens]
            seqlens = [int(x) for x in seqlens]
            return seqlens


def _lambda_mask_sparsity(seqlens: List[int], l_pretrain: int):
    n_starting = 64
    total_attn_elements = 0
    causal_attn_elements = 0
    for seqlen in seqlens:
        causal_attn_elements += seqlen * (seqlen + 1) // 2
        for i in range(seqlen):
            # first range, 0 to n_starting
            range1_start = 0
            range1_end = min(n_starting, i + 1)
            # second range, windowed with l_pretrain
            range2_start = max(0, i - l_pretrain + 1)
            range2_end = i + 1
            # get length of combination of two ranges
            range1_len = range1_end - range1_start
            range2_len = range2_end - range2_start
            # overlap
            overlap_start = max(range1_start, range2_start)
            overlap_end = min(range1_end, range2_end)
            overlap_len = max(0, overlap_end - overlap_start)
            total_attn_elements += range1_len + range2_len - overlap_len
    return total_attn_elements / causal_attn_elements


def _shared_question_mask_sparsity(seqlens: List[int], n_answers: int):
    question_proportion: float = 0.2
    total_attn_elements = 0
    causal_attn_elements = 0
    for seqlen in seqlens:
        causal_attn_elements += seqlen * (seqlen + 1) // 2
        n_question_tokens = int(seqlen * question_proportion)
        n_answer_tokens = seqlen - n_question_tokens
        per_answer_tokens = n_answer_tokens // n_answers
        for i in range(seqlen):
            if i < n_question_tokens:
                # normal causal mask
                total_attn_elements += i + 1
            else:
                # full mask on question tokens, causal mask on current answer
                total_attn_elements += n_question_tokens
                if (
                    i
                    < (per_answer_tokens * (n_answers - 1)) + n_question_tokens
                ):
                    answer_start = (
                        (i - n_question_tokens)
                        // per_answer_tokens
                        * per_answer_tokens
                        + n_question_tokens
                    )
                else:
                    # last answer
                    answer_start = (
                        n_question_tokens + (n_answers - 1) * per_answer_tokens
                    )
                answer_end = i + 1
                total_attn_elements += answer_end - answer_start
    return total_attn_elements / causal_attn_elements


def _causal_blockwise_mask_sparsity(seqlens: List[int], k_local: int):
    block_size: int = 256
    total_attn_elements = 0
    causal_attn_elements = 0
    for seqlen in seqlens:
        causal_attn_elements += seqlen * (seqlen + 1) // 2
        for i in range(seqlen):
            block_id = i // block_size
            n_blocks = (seqlen + block_size - 1) // block_size
            if block_id == 0:
                # first block, causal
                total_attn_elements += i + 1
            elif block_id == n_blocks - 1:
                # attend to all previous blocks
                prev_block_end = (block_id - 1) * block_size
                total_attn_elements += prev_block_end
                # causal within this block
                total_attn_elements += i - prev_block_end + 1
            else:
                # attend to first block
                total_attn_elements += block_size
                # attend to previous k_local blocks
                prev_block_end = max(block_size, i - k_local * block_size)
                total_attn_elements += (i + 1) - prev_block_end
    return total_attn_elements / causal_attn_elements


def get_reference_comm_volume_and_mask_sparsity(exp_dir: str):
    if os.path.isdir(os.path.join(exp_dir, "compiler")):
        debug_dir = os.path.join(exp_dir, "compiler")
    else:
        assert os.path.isdir(exp_dir)
        debug_dir = exp_dir

    (
        _,
        _,
        _,
        _,
        _,
        mask,
        lambda_lpretrain,
        shared_question_nans,
        causal_blockwise_klocal,
    ) = parse_exp_config(exp_dir)

    logs = os.listdir(debug_dir)
    logs = [log for log in logs if log.endswith(".log")]
    logs.sort()

    mean_comms = []
    for log in logs:
        try:
            seqlens = _parse_n_tokens(os.path.join(debug_dir, log))
        except Exception as e:
            iter_id = int(log.split(".")[0].split("_")[-1][4:])
            if iter_id < 200:
                print(f"Error parsing log {log}: {e}")
            continue
        n_tokens = sum(seqlens)
        # round up to 2 * CP_SIZE = 2 * 2 * RING_SIZE
        n_tokens = (
            (n_tokens + 4 * RING_SIZE - 1) // (4 * RING_SIZE) * (4 * RING_SIZE)
        )
        n_tokens_per_node = n_tokens // RING_SIZE
        total_comm = TOTAL_COMM_PER_TOKEN * n_tokens_per_node / 1e6  # in MB
        mean_comms.append(total_comm)
        if mask == "causal":
            sparsity = 1.0
        elif mask == "lambda":
            sparsity = _lambda_mask_sparsity(seqlens, lambda_lpretrain)
        elif mask == "shared_question":
            sparsity = _shared_question_mask_sparsity(
                seqlens, shared_question_nans
            )
        elif mask == "causal_blockwise":
            sparsity = _causal_blockwise_mask_sparsity(
                seqlens, causal_blockwise_klocal
            )
        else:
            raise ValueError(f"Unknown mask type: {mask}")
    mean_comm = sum(mean_comms) / len(mean_comms)
    return mean_comm, sparsity


def parse_commops(input_string):
    """
    Parses a string containing multiple CommOp entries into a list of tuples.

    Args:
        input_string (str): A string containing CommOp entries separated by commas.

    Returns:
        list: A list of tuples, where each tuple contains the parsed components of a CommOp.
    """
    entries = []
    i = 0
    n = len(input_string)

    while i < n:
        # Find the start of a CommOp entry
        start = input_string.find("CommOp(", i)
        if start == -1:
            break  # No more entries

        # Track parentheses balance to find the end of the entry
        balance = 1
        end = start + len("CommOp(")
        while end < n and balance > 0:
            if input_string[end] == "(":
                balance += 1
            elif input_string[end] == ")":
                balance -= 1
            end += 1

        if balance != 0:
            raise ValueError("Unbalanced parentheses in input string")

        # Extract the full CommOp entry
        entry_str = input_string[start:end].strip()
        entries.append(entry_str)
        i = end  # Move to the end of the current entry

    result = []
    for entry in entries:
        # Remove 'CommOp(' prefix and closing ')'
        content = entry[len("CommOp(") : -1].strip()
        comm_type = content.split("comm_type=")[1].split(",", 1)[0].strip()
        peer = eval(content.split("peer=")[1].split("),", 1)[0].strip() + ")")
        buffer_block = (
            content.split("buffer_block=")[1].strip()[1:-1].split(",")
        )
        buffer_type = buffer_block[0].strip()
        buffer_size = int(buffer_block[1].strip())
        n_tokens = int(buffer_block[2].strip())
        buffer_block = (buffer_type, buffer_size, n_tokens)

        result.append((comm_type, peer, buffer_block))

    return result


def parse_log(log):
    with open(log, "r") as f:
        lines = f.readlines()
    lines = [line.strip() for line in lines]
    lines = [line for line in lines if line]
    block_size = None
    inter_node_cost = None
    per_device_memory = []
    per_device_compute = []
    max_memory_per_device = None
    compilation_time = None
    for line in lines:
        if "Using block size:" in line:
            block_size = int(
                line.split("Using block size:")[1].strip().split(",")[0]
            )
        if "Inter-node cost:" in line:
            inter_node_cost = float(
                line.split("Inter-node cost:")[1].strip().split(" ")[0]
            )
        if "(total mem:" in line:
            memory = (
                int(line.split("(total mem:")[1].strip().split(")")[0]) / 1e6
            )
            per_device_memory.append(memory)
        if "(total workload:" in line:
            compute = float(
                line.split("(total workload:")[1].strip().split(")")[0]
            )
            per_device_compute.append(compute)
        if "Compilation took" in line:
            compilation_time = float(
                line.split("Compilation took")[1].strip().split(" ")[0]
            )
    assert block_size is not None
    assert inter_node_cost is not None
    assert compilation_time is not None
    assert len(per_device_memory) > 0
    max_memory_per_device = max(per_device_memory)
    assert len(per_device_compute) > 0
    compute_imbalance = max(per_device_compute) / (
        sum(per_device_compute) / len(per_device_compute)
    )
    return (
        block_size,
        inter_node_cost,
        max_memory_per_device,
        compute_imbalance,
        compilation_time,
    )


def _get_buffer_per_device_size(buffer_name: str, n_tokens: int):
    if "Q" in buffer_name or "Out" in buffer_name:
        # shape (n_tokens, 1, 128) since we use head block size = 1
        return n_tokens * 128 * 2
    elif "KV" in buffer_name:
        return 2 * n_tokens * 128 * 2
    elif "LSE" in buffer_name:
        return 4 * n_tokens  # fp32
    else:
        raise ValueError(f"Unknown buffer name: {buffer_name}")


def _parse_per_device_internode_send_recv(log):
    with open(log, "r") as f:
        lines = f.readlines()
    lines = [line.strip() for line in lines]
    lines = [line for line in lines if line]
    exec_plan_started = False
    curr_device = None
    per_device_internode_sends = defaultdict(int)
    per_device_internode_recvs = defaultdict(int)
    for line in lines:
        if "Forward Execution Plan:" in line:
            exec_plan_started = True
        if exec_plan_started:
            if "Device" in line:
                curr_device = tuple(
                    int(x.strip())
                    for x in line.split("Device")[1]
                    .split(":")[0]
                    .strip()[1:-1]
                    .split(",")
                )
            if "CommLaunchInstr" in line:
                comm_ops_str = line.split("comm_ops=[")[1].split("], stream=")[
                    0
                ]
                comm_ops = parse_commops(comm_ops_str)
                for comm_op in comm_ops:
                    comm_type, peer, buffer_block = comm_op
                    if peer[0] != curr_device[0]:
                        buffer_type, _, n_tokens = buffer_block
                        buffer_size = _get_buffer_per_device_size(
                            buffer_type, n_tokens
                        )
                        if comm_type == "send":
                            per_device_internode_sends[
                                curr_device
                            ] += buffer_size
                        elif comm_type == "recv":
                            per_device_internode_recvs[
                                curr_device
                            ] += buffer_size
        if "Backward Execution Plan:" in line:
            exec_plan_started = False
            break
    max_sends = max(per_device_internode_sends.values())
    max_recvs = max(per_device_internode_recvs.values())
    return max(max_sends, max_recvs)


def get_max_comm_volume_after_schedule(exp_dir: str):
    if os.path.isdir(os.path.join(exp_dir, "compiler")):
        debug_dir = os.path.join(exp_dir, "compiler")
    else:
        assert os.path.isdir(exp_dir)
        debug_dir = exp_dir

    logs = os.listdir(debug_dir)
    logs = [log for log in logs if log.endswith(".log")]
    logs.sort()

    mean_comms = []
    for log in logs:
        try:
            max_comm = _parse_per_device_internode_send_recv(
                os.path.join(debug_dir, log)
            )
        except Exception as e:
            iter_id = int(log.split(".")[0].split("_")[-1][4:])
            if iter_id < 200:
                print(f"Error parsing log {log}: {e}")
            continue
        mean_comms.append(max_comm)
    mean_comm = sum(mean_comms) / len(mean_comms)
    return mean_comm


def parse_exp_config(exp_dir: str):
    exp_dir = os.path.basename(exp_dir)
    dcp_block_size = None
    mem_imbal = None
    compute_imbal = None
    internode_comp_imbal_factor = None
    mask = None
    lambda_lpretrain = -1
    shared_question_nans = -1
    causal_blockwise_klocal = -1
    try:
        _, _, _, bsz_str, _, msl_str, _, bal_str, rest = exp_dir.split("_", 8)
    except:
        print(f"Error parsing exp_dir: {exp_dir}")
        raise
    msl = int(msl_str.split("msl")[1])
    dcp_block_size = int(bsz_str.split("bsz")[1])
    mem_imbal, compute_imbal, internode_comp_imbal_factor = (
        float(x) for x in bal_str[3:].split(",")
    )
    rest = rest[3:]
    if "causal_blockwise" in rest:
        mask = "causal_blockwise"
        if "klocal" in rest:
            causal_blockwise_klocal = int(rest.split("klocal")[1])
    elif "shared_question" in rest:
        mask = "shared_question"
        if "nans" in rest:
            shared_question_nans = int(rest.split("nans")[1])
    elif "lambda" in rest:
        mask = "lambda"
        if "lpretrain" in rest:
            lambda_lpretrain = int(rest.split("lpretrain")[1])
    else:
        mask = "causal"
    return (
        dcp_block_size,
        msl,
        mem_imbal,
        compute_imbal,
        internode_comp_imbal_factor,
        mask,
        lambda_lpretrain,
        shared_question_nans,
        causal_blockwise_klocal,
    )


def preprocess_exp_dir(
    root_dir: str, out_dir: str, dataset: str, node_id: int
):
    df_data = []
    exp_dirs = [
        d
        for d in os.listdir(os.path.join(root_dir, dataset))
        if not d.startswith(".")
    ]
    exp_dirs = [os.path.join(root_dir, dataset, d) for d in exp_dirs]
    exp_configs = [parse_exp_config(d) for d in exp_dirs]
    for i, exp_dir in tqdm(
        enumerate(exp_dirs), total=len(exp_dirs), desc="Experiments"
    ):
        (
            dcp_block_size,
            msl,
            mem_imbal,
            compute_imbal,
            internode_comp_imbal_factor,
            mask,
            lambda_lpretrain,
            shared_question_nans,
            causal_blockwise_klocal,
        ) = exp_configs[i]
        (
            _,
            mean_internode_cost,
            max_memory_per_device,
            mean_compute_imbalance,
            mean_compilation_time,
        ) = parse_results(exp_dir)
        # get reference comm volume
        ref_internode_cost, mask_sparsity = (
            get_reference_comm_volume_and_mask_sparsity(exp_dir)
        )
        # get max comm volume after schedule
        max_internode_cost_after_schedule = get_max_comm_volume_after_schedule(
            exp_dir
        )
        df_data.append(
            {
                "dataset": dataset,
                "max_seq_len": msl,
                "dcp_block_size": dcp_block_size,
                "mem_imbal": mem_imbal,
                "compute_imbal": compute_imbal,
                "internode_comp_imbal_factor": internode_comp_imbal_factor,
                "mask": mask,
                "lambda_lpretrain": lambda_lpretrain,
                "shared_question_nans": shared_question_nans,
                "causal_blockwise_klocal": causal_blockwise_klocal,
                "mean_internode_cost": mean_internode_cost,
                "max_memory_per_device": max_memory_per_device,
                "mean_compute_imbalance": mean_compute_imbalance,
                "mean_compilation_time": mean_compilation_time,
                "ref_internode_cost": ref_internode_cost,
                "max_internode_cost_after_schedule": max_internode_cost_after_schedule,
                "mask_sparsity": mask_sparsity,
            }
        )
    df = pd.DataFrame(df_data)
    filename = f"dryrun_results_N{node_id}.csv"
    df.to_csv(os.path.join(out_dir, filename), index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess dryrun results")
    parser.add_argument(
        "--exp-dir", type=str, required=True, help="Experiment directory"
    )
    parser.add_argument(
        "--out-dir", type=str, required=True, help="Output directory"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="LongAlign",
        choices=["LongAlign", "LDC"],
        help="Dataset to plot.",
    )
    parser.add_argument("--node-id", type=int, required=True, help="Node ID")

    args = parser.parse_args()

    if args.dataset == "LongAlign":
        args.dataset = "THUDM_LongAlign-10k"
    elif args.dataset == "LDC":
        args.dataset = "jchenyu_Long-Data-Collections-sample-10000"
    else:
        raise ValueError(
            f"Unsupported dataset: {args.dataset}."
            "Should be one of 'LongAlign' and 'LDC'."
        )

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    preprocess_exp_dir(args.exp_dir, args.out_dir, args.dataset, args.node_id)
