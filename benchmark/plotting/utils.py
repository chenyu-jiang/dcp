import os
from dataclasses import dataclass
import json


def get_log_path(log_dir):
    return os.path.join(log_dir, "stdout_stderr.log")


def parse_throughput_results(log_dir):
    iteration_times = []
    with open(get_log_path(log_dir), "r") as f:
        for line in f:
            if "elapsed time per iteration" in line:
                t = float(
                    line.split("elapsed time per iteration (ms):")[-1]
                    .strip()
                    .split()[0]
                )
                iteration_times.append(t)
    return iteration_times[2:]  # skip first 20 warmup iterations


def parse_loss_curve(log_dir):
    losses_per_iter = []
    with open(get_log_path(log_dir), "r") as f:
        for line in f:
            if "Training loss:" in line:
                t = float(line.split("Training loss:")[-1].strip())
                losses_per_iter.append(t)
    return losses_per_iter


def get_dataset_abbreviation(dataset_name):
    if dataset_name == "THUDM_LongAlign-10k":
        return "LongAlign"
    elif dataset_name == "jchenyu_Long-Data-Collections-sample-10000":
        return "LDC"
    else:
        return dataset_name


@dataclass(eq=True)
class ExperimentConfig:
    max_seq_len: int = 0
    n_tokens_per_global_batch: int = 0
    block_size: int = 0
    head_block_size: int = 1
    mask_type: str = "causal"
    tp_size: int = 1
    cp_size: int = 1
    dp_size: int = 1
    mem_imbalance_epsilon: float = 0.0
    comp_imbalance_epsilon: float = 0.0
    inter_node_comp_imbalance_factor: float = 0.0
    status: str = "unknown"

    @staticmethod
    def parse_experiment_status(exp_dir):
        log_path = os.path.join(exp_dir, "stdout_stderr.log")
        args_path = os.path.join(exp_dir, "args.json")
        if not os.path.exists(log_path) or not os.path.exists(args_path):
            print(f"Log file or args file not found in {exp_dir}")
            return "unknown"
        with open(args_path, "r") as f:
            config_args = json.load(f)
            n_iters = config_args["n_iters"]
        with open(log_path, "r") as f:
            contents = f.read()
        if "after training is done" in contents or "StopIteration" in contents:
            return "success"
        elif contents.count("Training loss:") >= n_iters:
            return "success"
        elif (
            "AssertionError" in contents
            or "OutOfMemoryError" in contents
            or "RuntimeError" in contents
            or "out of memory" in contents
            or "Connection reset by peer" in contents
        ):
            return "failure"
        return "unknown"

    @staticmethod
    def parse_history_experiments(exp_dir):
        exp_spec = os.path.basename(os.path.normpath(exp_dir))
        config_items = exp_spec.split("_", 8)
        config = ExperimentConfig()
        for item in config_items:
            if item.startswith("dp"):
                config.dp_size = int(item[2:])
            elif item.startswith("tp"):
                config.tp_size = int(item[2:])
            elif item.startswith("cp"):
                config.cp_size = int(item[2:])
            elif item.startswith("bsz"):
                config.block_size = int(item[3:])
            elif item.startswith("hbs"):
                config.head_block_size = int(item[3:])
            elif item.startswith("msl"):
                config.max_seq_len = int(item[3:])
            elif item.startswith("gbs"):
                config.n_tokens_per_global_batch = int(item[3:])
            elif item.startswith("bal"):
                balance_vals = item[3:].split(",")
                config.mem_imbalance_epsilon = float(balance_vals[0])
                config.comp_imbalance_epsilon = float(balance_vals[1])
                config.inter_node_comp_imbalance_factor = float(
                    balance_vals[2]
                )
            elif item.startswith("msk"):
                config.mask_type = item[3:]
        # test status
        config.status = ExperimentConfig.parse_experiment_status(exp_dir)
        return config
