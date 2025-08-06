import argparse
import json
import os
import sys
import pwd
import time
from functools import partial
import subprocess
from dataclasses import dataclass
from string import Template

from benchmark.utils import RedisKVStore, kill_redis_servers

# PROFILE_CMD = "/usr/local/bin/nsys profile -w true -t cuda,nvtx,osrt,cudnn,cublas,python-gil -s none --python-sampling true -o {} -f true --cudabacktrace sync --capture-range=cudaProfilerApi -x true --capture-range-end stop"
PROFILE_CMD = "/usr/local/bin/nsys profile -w true -t cuda,nvtx,osrt,cudnn,cublas,python-gil -s none -o {} -f true --cudabacktrace sync --capture-range=cudaProfilerApi -x true --capture-range-end stop"
MODEL_CONFIGS = {
    "gpt2-4b": {
        "model_n_layers": 24,
        "model_hidden_size": 2048,
        "model_n_heads": 16,
        "model_n_query_groups": 4,
        "model_head_size": 128,
        "model_ffn_hidden_size": 8192,
    },
    "gpt2-8b": {
        "model_n_layers": 32,
        "model_hidden_size": 4096,
        "model_n_heads": 32,
        "model_n_query_groups": 8,
        "model_head_size": 128,
        "model_ffn_hidden_size": 14336,
    },
    "test": {
        "model_n_layers": 4,
        "model_hidden_size": 2048,
        "model_n_heads": 16,
        "model_n_query_groups": 8,
        "model_head_size": 128,
        "model_ffn_hidden_size": 8192,
    },
}

EXPERIMENT_DIR_PREFIX = "./experiments/"
EXPERIMENT_PROGRESS_TIMEOUT = 180  # 3 mins
EXPERIMENT_PROGRESS_REPORT_INTERVAL = 60  # 1 min
EXPERIMENT_PROGRESS_POLL_INTERVAL = 5  # 5s

print_fn = print


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

    def memory_dominates(self, other):
        assert isinstance(
            other, ExperimentConfig
        ), "Can only compare with ExperimentConfig"
        # Config A memory_dominates config B if A is almost surely
        # consumes more memory than B
        # if parallelism are different, no dominance
        if (
            self.tp_size != other.tp_size
            or self.cp_size != other.cp_size
            or self.dp_size != other.dp_size
            or self.mask_type != other.mask_type
        ):
            return False
        if (
            self.n_tokens_per_global_batch == other.n_tokens_per_global_batch
            and self.max_seq_len > other.max_seq_len
        ):
            return True
        if (
            self.max_seq_len == other.max_seq_len
            and self.n_tokens_per_global_batch
            > other.n_tokens_per_global_batch
        ):
            return True
        return False

    def parallelism_dominates(self, other):
        assert isinstance(
            other, ExperimentConfig
        ), "Can only compare with ExperimentConfig"
        # Config A parallelism_dominates config B if A uses higher degree of
        # data parallelism than B
        # if A can run without OOM, this means A is almost surely faster than B
        # if number of gpus are different, no dominance
        if (
            self.tp_size * self.cp_size * self.dp_size
            != other.tp_size * other.cp_size * other.dp_size
            or self.max_seq_len != other.max_seq_len
            or self.n_tokens_per_global_batch
            != other.n_tokens_per_global_batch
            or self.mask_type != other.mask_type
        ):
            return False
        if self.tp_size != other.tp_size:
            return False
        if self.dp_size > other.dp_size:
            return True
        return False

    def global_batch_size_dominates(self, other):
        assert isinstance(
            other, ExperimentConfig
        ), "Can only compare with ExperimentConfig"
        if (
            self.tp_size * self.cp_size * self.dp_size
            != other.tp_size * other.cp_size * other.dp_size
            or self.mask_type != other.mask_type
        ):
            return False
        if (
            self.tp_size != other.tp_size
            or self.max_seq_len != other.max_seq_len
        ):
            return False
        return self.n_tokens_per_global_batch > other.n_tokens_per_global_batch

    @staticmethod
    def parse_experiment_status(exp_dir):
        log_path = os.path.join(exp_dir, "stdout_stderr.log")
        args_path = os.path.join(exp_dir, "args.json")
        if not os.path.exists(log_path) or not os.path.exists(args_path):
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
        config_items = exp_spec.split("_")
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


def get_exp_spec_name(args):
    nranks = args.num_nodes * args.num_gpus_per_node
    dp_size = nranks // (args.tp_size * args.cp_size)
    exp_spec_name = "tp{}_dp{}_cp{}".format(
        args.tp_size, dp_size, args.cp_size
    )
    if args.dcp_use_block_size_heuristic or args.framework != "dcp":
        exp_spec_name += "_bsz-1"
    else:
        exp_spec_name += "_bsz{}".format(args.dcp_block_size)
    exp_spec_name += "_hbs{}".format(args.dcp_head_block_size)
    exp_spec_name += "_msl{}_gbs{}".format(
        args.max_seq_len, args.n_tokens_per_global_batch
    )
    if args.framework == "dcp":
        balance_vals = "{:.1f},{:.1f},{:.1f}".format(
            args.dcp_mem_imbalance_epsilon,
            args.dcp_comp_imbalance_epsilon,
            args.dcp_inter_node_comp_imbalance_factor,
        )
    else:
        balance_vals = "0,0,0"
    exp_spec_name += "_bal" + balance_vals
    exp_spec_name += "_msk{}".format(args.mask_type)
    if args.profile:
        exp_spec_name += "_profile"
    return exp_spec_name


def cleanup_mlm_job(args):
    os.system("pkill -9 -f 'pretrain_gpt'")


def _is_pow_of_2(n):
    return n & (n - 1) == 0


def _calculate_cp_dp_size(args, is_grid_run=False):
    # check
    total_gpus = args.num_nodes * args.num_gpus_per_node
    assert (
        total_gpus % args.tp_size == 0
    ), "Total number of GPUs must be divisible by tensor parallelism size"
    if args.framework == "dcp":
        # dcp use dp as dp + cp dimension
        yield 1, total_gpus // args.tp_size
    else:
        # calculate max dp size based on max seq len
        assert (
            args.n_tokens_per_global_batch % args.max_seq_len == 0
        ), "Global batch size must be divisible by max sequence length"
        max_dp_size = args.n_tokens_per_global_batch // args.max_seq_len
        if not is_grid_run:
            assert (
                total_gpus % (args.tp_size * args.cp_size) == 0
            ), "Total number of GPUs must be divisible by tensor parallelism size * context parallelism size"
            dp_size = total_gpus // (args.tp_size * args.cp_size)
            assert (
                dp_size <= max_dp_size
            ), f"Data parallelism size {dp_size} must be less than or equal to max data parallelism size {max_dp_size}"
            yield args.cp_size, dp_size
        else:
            # prioritize dp over cp
            max_dp_size = min(max_dp_size, total_gpus // args.tp_size)
            assert _is_pow_of_2(
                max_dp_size
            ), "Max data parallelism size should be a power of 2"
            while max_dp_size >= 1:
                dp_size = max_dp_size
                assert (
                    total_gpus % dp_size == 0
                ), "Total number of GPUs must be divisible by data parallelism size"
                assert (
                    total_gpus % (dp_size * args.tp_size) == 0
                ), "Total number of GPUs must be divisible by data parallelism size * tensor parallelism size"
                cp_size = total_gpus // (dp_size * args.tp_size)
                yield cp_size, dp_size
                max_dp_size //= 2


def _calculate_dcp_head_block_size(args, is_grid_run=False):
    if args.framework == "dcp":
        if not is_grid_run:
            yield args.dcp_head_block_size
        else:
            yield 1
    elif args.framework == "mlm_a2a_p2p":
        if not is_grid_run:
            yield args.dcp_head_block_size
        else:
            n_groups_per_tp_rank = (
                MODEL_CONFIGS[args.model]["model_n_query_groups"]
                // args.tp_size
            )
            if args.cp_size > n_groups_per_tp_rank:
                yield 1
            else:
                head_block_size = n_groups_per_tp_rank // args.cp_size
                yield head_block_size
    else:
        yield 1


def generate_exp_configs(args):

    if args.grid_run:
        # generate grid of max seq len and global batch size
        if args.grid_run_frameworks is not None:
            frameworks = args.grid_run_frameworks
        else:
            frameworks = ["dcp", "mlm_a2a_p2p"]
        mask_types = [
            "causal",
            "lambda",
            "shared_question",
            "causal_blockwise",
        ]
        max_seqlens = [16384, 32768, 65536, 131072]
        if args.best_config is not None:
            # load best configs from file
            with open(args.best_config, "r") as f:
                best_configs = json.load(f)
            # convert from list of dicts to set with key
            # (dataset, max_seq_len, global_batch_size,
            #  dcp_block_size, dcp_head_block_size, mask_type,
            #  tp_size, cp_size, dp_size,
            #  dcp_mem_imbalance_epsilon, dcp_comp_imbalance_epsilon,
            #  dcp_inter_node_comp_imbalance_factor)
            best_configs = set(
                (
                    config["Dataset"],
                    config["MaxSeqLen"],
                    config["GlobalBatchSize"],
                    config["BlockSize"],
                    config["HeadBlockSize"],
                    config["MaskType"],
                    config["TPSize"],
                    config["CPSize"],
                    config["DPSize"],
                    config["MemImbalanceEpsilon"],
                    config["CompImbalanceEpsilon"],
                    config["InterNodeCompImbalanceFactor"],
                )
                for config in best_configs
            )
        else:
            best_configs = None
        for framework in frameworks:
            args.framework = framework
            for max_seq_len in max_seqlens:
                # for global_batch_size in global_batch_sizes:
                for global_batch_size in [max_seq_len]:
                    if global_batch_size < max_seq_len:
                        print(
                            "Skip because global batch size {} < max seq len {}".format(
                                global_batch_size, max_seq_len
                            )
                        )
                        continue
                    args.max_seq_len = max_seq_len
                    args.n_tokens_per_global_batch = global_batch_size
                    if (
                        framework == "dcp"
                        and not args.dcp_use_block_size_heuristic
                    ):
                        if max_seq_len > 65536:
                            block_sizes = [2048, 4096]
                        elif max_seq_len <= 16384:
                            block_sizes = [512, 1024]
                        else:
                            block_sizes = [1024]
                    else:
                        block_sizes = [-1]  # unused
                    for block_size in block_sizes:
                        args.dcp_block_size = block_size
                        for cp_size, dp_size in _calculate_cp_dp_size(
                            args, is_grid_run=True
                        ):
                            args.cp_size = cp_size
                            args.dp_size = dp_size
                            for hbs in _calculate_dcp_head_block_size(
                                args, is_grid_run=True
                            ):
                                args.dcp_head_block_size = hbs
                                if framework == "dcp":
                                    comp_imbal_scale_eps = [4.0, 16.0]
                                else:
                                    comp_imbal_scale_eps = [0.0]  # unused
                                for (
                                    comp_imbal_scale_ep
                                ) in comp_imbal_scale_eps:
                                    args.dcp_inter_node_comp_imbalance_factor = (
                                        comp_imbal_scale_ep
                                    )
                                    for mask_type in mask_types:
                                        args.mask_type = mask_type
                                        if (
                                            args.framework != "dcp"
                                            and args.mask_type != "causal"
                                        ):
                                            args.dcp_prefetch_listener_num_workers = (
                                                12
                                            )
                                        else:
                                            args.dcp_prefetch_listener_num_workers = (
                                                2
                                            )
                                        # check best configs here
                                        if best_configs is not None:
                                            curr_exp_key = (
                                                args.dataset,
                                                args.max_seq_len,
                                                args.n_tokens_per_global_batch,
                                                args.dcp_block_size,
                                                args.dcp_head_block_size,
                                                args.mask_type,
                                                args.tp_size,
                                                args.cp_size,
                                                args.dp_size,
                                                args.dcp_mem_imbalance_epsilon,
                                                args.dcp_comp_imbalance_epsilon,
                                                args.dcp_inter_node_comp_imbalance_factor,
                                            )
                                            if (
                                                curr_exp_key
                                                not in best_configs
                                            ):
                                                print_fn(
                                                    f"Skip because exp is not in the best configs."
                                                )
                                                continue
                                        yield args, ExperimentConfig(
                                            args.max_seq_len,
                                            args.n_tokens_per_global_batch,
                                            (
                                                args.dcp_block_size
                                                if framework == "dcp"
                                                else -1
                                            ),
                                            args.dcp_head_block_size,
                                            args.mask_type,
                                            args.tp_size,
                                            args.cp_size,
                                            args.dp_size,
                                            (
                                                args.dcp_mem_imbalance_epsilon
                                                if framework == "dcp"
                                                else 0.0
                                            ),
                                            (
                                                args.dcp_comp_imbalance_epsilon
                                                if framework == "dcp"
                                                else 0.0
                                            ),
                                            (
                                                args.dcp_inter_node_comp_imbalance_factor
                                                if framework == "dcp"
                                                else 0.0
                                            ),
                                        )
    else:
        assert (
            args.mask_type is not None
        ), "mask_type must be specified for non-grid run"
        if args.framework != "dcp" and args.mask_type != "causal":
            args.dcp_prefetch_listener_num_workers = 12
        else:
            args.dcp_prefetch_listener_num_workers = 2
        for cp_size, dp_size in _calculate_cp_dp_size(args):
            args.cp_size = cp_size
            args.dp_size = dp_size
            for hbs in _calculate_dcp_head_block_size(args):
                args.dcp_head_block_size = hbs
                yield args, ExperimentConfig(
                    args.max_seq_len,
                    args.n_tokens_per_global_batch,
                    args.dcp_block_size,
                    args.dcp_head_block_size,
                    args.mask_type,
                    args.tp_size,
                    args.cp_size,
                    args.dp_size,
                    args.dcp_mem_imbalance_epsilon,
                    args.dcp_comp_imbalance_epsilon,
                    args.dcp_inter_node_comp_imbalance_factor,
                )


def _check_logging_args(args):
    exp_spec_name = get_exp_spec_name(args)
    exp_logging_dir = os.path.join(
        EXPERIMENT_DIR_PREFIX,
        args.dataset.replace("/", "_"),
        args.framework,
        exp_spec_name,
    )
    if os.path.exists(exp_logging_dir):
        # exp dir already exists
        return args, exp_logging_dir, True
    if not os.path.exists(exp_logging_dir):
        os.makedirs(exp_logging_dir)
    args.stdout_stderr_log = os.path.join(exp_logging_dir, "stdout_stderr.log")
    # dump all args to a file
    args_file = os.path.join(exp_logging_dir, "args.json")
    with open(args_file, "w") as f:
        args_dict = vars(args)
        dump_dict = {k: v for k, v in args_dict.items() if k != "kvstore"}
        json.dump(dump_dict, f, indent=2)
    return args, exp_logging_dir, False


def _get_shell_script(args, exp_logging_dir):
    if args.template is None:
        args.template = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "launch_template.txt"
        )

    model_config = MODEL_CONFIGS[args.model]

    # check
    total_gpus = args.num_nodes * args.num_gpus_per_node
    assert (
        total_gpus % args.tp_size == 0
    ), "Total number of GPUs must be divisible by tensor parallelism size"
    # calculate remaining identifiers based on args
    user_home = os.path.expanduser("~")
    dcp_args = (
        f"--dcp-global-batch-size-in-tokens {args.n_tokens_per_global_batch} "
    )
    dcp_args += f"--dcp-dataset-split {args.dataset_split} "
    dcp_args += f"--dcp-dataset-text-key {args.dataset_text_key} "
    dcp_args += f"--dcp-mask-type {args.mask_type} "
    dcp_args += f"--dcp-prefetch-listener-num-workers {args.dcp_prefetch_listener_num_workers} "
    if args.framework == "dcp":
        if args.dcp_use_block_size_heuristic:
            dcp_args += "--dcp-use-block-size-heuristic "
        dcp_args += f"--dcp-block-size {args.dcp_block_size} "
        dcp_args += f"--dcp-head-block-size {args.dcp_head_block_size} "
        dcp_args += (
            f"--dcp-mem-imbalance-epsilon {args.dcp_mem_imbalance_epsilon} "
        )
        dcp_args += (
            f"--dcp-comp-imbalance-epsilon {args.dcp_comp_imbalance_epsilon} "
        )
        dcp_args += f"--dcp-inter-node-comp-imbalance-factor {args.dcp_inter_node_comp_imbalance_factor} "
        if args.cudagraph:
            dcp_args += "--dcp-use-cudagraph "
    else:
        dcp_args += "--disable-dcp "
    logging_dir_full_path = os.path.abspath(exp_logging_dir)
    if args.profile:
        profile_out_file = os.path.join(
            logging_dir_full_path, "nsys_report.nsys-rep"
        )
        if args.profiler == "nsys":
            profile_cmd = PROFILE_CMD.format(profile_out_file)
        else:
            profile_cmd = ""
        profile_args = (
            "--profile --profile-step-start 50 --profile-step-end 55 --profile-ranks "
            + " ".join([str(x) for x in range(total_gpus)])
            + " "
        )
        if args.profiler == "pytorch":
            tensorboard_dir = os.path.join(
                logging_dir_full_path, "tensorboard"
            )
            if not os.path.exists(tensorboard_dir):
                os.makedirs(tensorboard_dir)
            profile_args += (
                f"--use-pytorch-profiler --tensorboard-dir {tensorboard_dir} "
            )
    else:
        profile_cmd = ""
        profile_args = ""
    gqa_args = ""
    if model_config["model_n_heads"] != model_config["model_n_query_groups"]:
        gqa_args = f"--group-query-attention "
    mlm_cp_args = ""
    if args.framework in ["mlm_p2p", "mlm_a2a_p2p"]:
        mlm_cp_args += (
            f"--cp-comm-type {'+'.join(args.framework.split('_')[1:])} "
        )
        if args.framework == "mlm_a2a_p2p":
            n_query_groups = (
                model_config["model_n_query_groups"] // args.tp_size
            )
            a2a_size = n_query_groups // args.dcp_head_block_size
            p2p_size = args.cp_size // a2a_size
            mlm_cp_args += (
                f"--hierarchical-context-parallel-sizes {a2a_size} {p2p_size}"
            )
    template_args = {
        "profile_cmd": profile_cmd,
        "host_ip": args.host_ip,
        "hf_home": args.hf_home,
        "user_home": user_home,
        "dcp_log_schedule": "1" if args.dcp_log_schedule else "0",
        "dcp_log_executor": "1" if args.dcp_log_executor else "0",
        "n_gpus_per_node": args.num_gpus_per_node,
        "n_nodes": args.num_nodes,
        "node_rank": args.node_rank,
        "tp_size": args.tp_size,
        "cp_size": args.cp_size,
        "dp_size": args.dp_size,
        "max_seq_len": args.max_seq_len,
        "n_iters": args.n_iters,
        "dataset_path": args.dataset,
        "log_interval": args.log_interval,
        "dcp_args": dcp_args,
        "profile_args": profile_args,
        "stdout_stderr_log": args.stdout_stderr_log,
        "logging_dir": logging_dir_full_path,
        "gqa_args": gqa_args,
        "mlm_cp_args": mlm_cp_args,
        "init_dataset_with_global_rank": args.init_dataset_with_global_rank,
    } | model_config
    # read template
    with open(args.template) as f:
        launch_template = Template(f.read())
        launch_script = launch_template.substitute(template_args)
    return launch_script


def run_grid_experiments(args):
    global print_fn
    config_iterator = generate_exp_configs(args)
    from tqdm import tqdm

    is_tty = sys.__stdin__.isatty()

    config_iterator = tqdm(config_iterator, disable=None)
    print_fn = config_iterator.write if is_tty else partial(print, flush=True)
    for current_args, current_exp_config in config_iterator:
        current_exp_config: ExperimentConfig
        past_success_configs = []
        past_failures_configs = []
        exp_dir = os.path.join(
            EXPERIMENT_DIR_PREFIX,
            args.dataset.replace("/", "_"),
            args.framework,
        )
        if os.path.isdir(exp_dir):
            for exp_spec_dir in [
                os.path.join(exp_dir, x)
                for x in os.listdir(exp_dir)
                if os.path.isdir(os.path.join(exp_dir, x))
            ]:
                config = ExperimentConfig.parse_history_experiments(
                    exp_spec_dir
                )
                if config.status == "success":
                    past_success_configs.append(config)
                else:
                    past_failures_configs.append(config)
        should_skip = False
        for past_success_config in past_success_configs:
            past_success_config: ExperimentConfig
            pass
            # if past_success_config.global_batch_size_dominates(
            #     current_exp_config
            # ):
            #     print_fn(
            #         f"Skip {current_exp_config} because succeeded config {past_success_config} has larger global batch size."
            #     )
            #     should_skip = True
            #     break
            # if past_success_config.parallelism_dominates(current_exp_config):
            #     print_fn(
            #         f"Skip {current_exp_config} because succeeded config {past_success_config} has higher degree of DP."
            #     )
            #     should_skip = True
            #     break
        for past_failure_config in past_failures_configs:
            past_failure_config: ExperimentConfig
            pass
            # if current_exp_config.memory_dominates(past_failure_config):
            #     print_fn(
            #         f"Skip {current_exp_config} because it consumes more memory than {past_failure_config}"
            #     )
            #     should_skip = True
            #     break
        assert hasattr(args, "kvstore") and args.kvstore is not None
        print_fn("Preparing experiment in {}...".format(exp_dir))
        kv: RedisKVStore = args.kvstore
        # synchronize should skip
        print_fn("Gathering status before args checking...")
        should_skips = kv.allgather(should_skip)
        should_skip = any(should_skips)
        if should_skip:
            print_fn("Skipping experiment.")
            continue
        current_args, exp_logging_dir, should_skip = _check_logging_args(
            current_args
        )
        print_fn("Gathering status after args checking...")
        should_skips = kv.allgather(should_skip)
        should_skip = any(should_skips)
        spec_basename = os.path.basename(exp_logging_dir)
        if is_tty:
            config_iterator.set_description(
                "Running experiment {}".format(spec_basename)
            )
        else:
            print_fn("Running experiment {}".format(spec_basename))
        if should_skip:
            print_fn(
                "Skip {} because it has already been run.".format(
                    spec_basename
                )
            )
            # the experiment has been run
            continue
        # barrier before starting the experiment
        print_fn("Before syncing exp config...")
        kv.barrier()
        # exchange exp config
        gathered_exp_configs = kv.gather(current_exp_config)
        if kv.node_rank == 0:
            # check if all nodes have the same exp config
            if not all(
                [
                    gathered_exp_config == current_exp_config
                    for gathered_exp_config in gathered_exp_configs
                ]
            ):
                print(
                    "ERROR: All nodes must have the same experiment config, but got {}".format(
                        gathered_exp_configs
                    )
                )
                kv.send_abort_signal()
                sys.exit(1)
        kv.barrier()
        print_fn("Before generating shell script...")
        shell_script = _get_shell_script(current_args, exp_logging_dir)
        shell_script_path = os.path.join(exp_logging_dir, "run.sh")
        with open(shell_script_path, "w") as f:
            f.write(shell_script)
        # all stdout and stderr are redirected to current_args.stdout_stderr_log
        if args.dry_run:
            continue
        p = subprocess.Popen(
            f"bash {shell_script_path}",
            shell=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        print_fn("Process started...")
        last_progress_report_time = time.time()

        assert (
            current_args.stdout_stderr_log is not None
        ), "stdout_stderr_log must be specified for batch experiments."
        prev_content = None
        last_progress = time.time()
        while p.poll() is None:
            should_stop = False
            has_progress = True
            # check if the stdout/stderr log has progress
            if not os.path.exists(current_args.stdout_stderr_log):
                # the job has not started yet
                if time.time() - last_progress > EXPERIMENT_PROGRESS_TIMEOUT:
                    has_progress = False
                else:
                    time.sleep(EXPERIMENT_PROGRESS_POLL_INTERVAL)
                    continue
            with open(current_args.stdout_stderr_log, "r") as f:
                current_content = f.read()
            if (
                "AssertionError" in current_content
                or "OutOfMemoryError" in current_content
                or "RuntimeError" in current_content
                or "out of memory" in current_content
                or "IndexError" in current_content
                or "ERROR" in current_content
                or "SIGTERM" in current_content
                or "Traceback (most recent call last):" in current_content
            ):
                # error
                should_stop = True
            elif (
                "after training is done" in current_content
                or "process group has NOT been destroyed" in current_content
            ):
                # finished
                should_stop = True
            elif current_content != prev_content:
                # progress
                prev_content = current_content
                last_progress = time.time()
            else:
                # no progress
                if time.time() - last_progress > EXPERIMENT_PROGRESS_TIMEOUT:
                    # timeout
                    has_progress = False
            if should_stop:
                kv.set(spec_basename + f"status_{args.node_rank}", "abort")
            kv.set(
                spec_basename + f"progress_{args.node_rank}", str(has_progress)
            )
            # get the most updated status from all nodes
            current_status = None
            for i in range(args.num_nodes):
                node_status = kv.get(spec_basename + f"status_{i}")
                if node_status is not None and not isinstance(
                    node_status, str
                ):
                    node_status = node_status.decode()
                if current_status is None:
                    current_status = node_status
                elif node_status == "abort":
                    current_status = "abort"
            progress_across_nodes = []
            for i in range(args.num_nodes):
                node_progress = kv.get(spec_basename + f"progress_{i}")
                if node_progress is not None and not isinstance(
                    node_progress, str
                ):
                    node_progress = node_progress.decode()
                if node_progress == "False":
                    progress_across_nodes.append(False)
                else:
                    progress_across_nodes.append(True)
            if (
                time.time() - last_progress_report_time
                > EXPERIMENT_PROGRESS_REPORT_INTERVAL
            ):
                last_progress_report_time = time.time()
                print_fn(
                    f"Experiment {spec_basename} progress across nodes: {progress_across_nodes}"
                )
            all_no_progress = all([not x for x in progress_across_nodes])
            should_stop = current_status == "abort" or all_no_progress
            if current_status == "abort":
                print_fn(
                    f"Abort signal received for config {current_exp_config}"
                )
            if all_no_progress:
                print_fn(
                    f"No progress for config {current_exp_config}, aborting."
                )
            if should_stop:
                # kill the job on all nodes
                if p.poll() is None:
                    p.kill()
                cleanup_mlm_job(args)
                kill_redis_servers(args.node_rank, args.kvstore)
                break
            # check if the entire script needs to abort
            if kv.check_abort_signal():
                sys.exit(1)
            time.sleep(EXPERIMENT_PROGRESS_POLL_INTERVAL)
        kv.barrier()
        print_fn("Proceeding to next experiment...")
        # check current experiment status
        current_exp_config.status = ExperimentConfig.parse_experiment_status(
            exp_logging_dir
        )
        # exchange exp status
        # gathered_exp_status = kv.gather(current_exp_config.status)
        # if kv.node_rank == 0:
        #     # check if all nodes have the same exp config
        #     if not all(
        #         [
        #             status == current_exp_config.status
        #             for status in gathered_exp_status
        #         ]
        #     ):
        #         print(
        #             "ERROR: All nodes must have the same experiment status, but got {}".format(
        #                 gathered_exp_status
        #             )
        #         )
        #         kv.send_abort_signal()
        #         sys.exit(1)
        if current_exp_config.status == "success":
            past_success_configs.append(current_exp_config)
        elif current_exp_config.status == "failure":
            past_failures_configs.append(current_exp_config)
        # else:
        #     raise ValueError(
        #         f"Failed to parse experiment status: {exp_logging_dir}: {current_exp_config.status}"
        #     )
    args.kvstore.barrier()
    if args.kvstore.is_master:
        kill_redis_servers(
            args.node_rank, args.kvstore, include_controller=True
        )


def _parse_args():
    parser = argparse.ArgumentParser(description="Run MLM experiments")
    parser.add_argument(
        "--template",
        type=str,
        help="The shell script template file to use for launching the experiment",
    )
    parser.add_argument(
        "-f",
        "--framework",
        type=str,
        default="dcp",
        choices=["dcp", "mlm_p2p", "mlm_a2a_p2p"],
    )
    parser.add_argument(
        "--num-nodes",
        type=int,
        default=1,
        help="The number of nodes to run the experiment on",
    )
    parser.add_argument(
        "--num-gpus-per-node",
        type=int,
        default=8,
        help="The number of GPUs per node",
    )
    parser.add_argument(
        "--node-rank",
        type=int,
        default=0,
        help="The rank of the node",
    )
    parser.add_argument(
        "-ip", "--host-ip", type=str, default="localhost", help="The host IP"
    )
    parser.add_argument(
        "--hf-home",
        type=str,
        default="~/.cache/huggingface",
        help="The home directory of Hugging Face",
    )
    parser.add_argument("--tp-size", type=int, default=4)
    parser.add_argument("--cp-size", type=int)
    parser.add_argument("--max-seq-len", type=int, default=16384)
    parser.add_argument("--n-tokens-per-global-batch", type=int, default=32768)
    parser.add_argument("--n-iters", type=int, default=10000000000)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--dataset", type=str, default="THUDM/LongAlign-10k")
    parser.add_argument("--dataset-split", type=str, default="train")
    parser.add_argument("--dataset-text-key", type=str, default="messages")
    parser.add_argument("--dcp-head-block-size", type=int, default=1)
    parser.add_argument(
        "--model",
        type=str,
        default="gpt2-8b",
        choices=["gpt2-4b", "gpt2-8b", "test"],
    )
    parser.add_argument(
        "--mask-type",
        type=str,
        default="causal",
        choices=[
            "causal",
            "lambda",
            "shared_question",
            "causal_blockwise",
            "modality_specific",
            "modality_specific_sparse",
        ],
    )
    parser.add_argument("-g", "--cudagraph", action="store_true")
    parser.add_argument(
        "--profile", action="store_true", help="Profile with nsight systems"
    )
    parser.add_argument(
        "--profiler", type=str, default="nsys", choices=["nsys", "pytorch"]
    )
    parser.add_argument("--dcp-mem-imbalance-epsilon", type=float, default=0.2)
    parser.add_argument(
        "--dcp-comp-imbalance-epsilon", type=float, default=0.1
    )
    parser.add_argument(
        "--dcp-inter-node-comp-imbalance-factor",
        type=float,
        default=4.0,
    )
    parser.add_argument(
        "--dcp-prefetch-listener-num-workers", type=int, default=2
    )
    parser.add_argument("--dcp-block-size", type=int, default=1024)
    parser.add_argument("--dcp-use-block-size-heuristic", action="store_true")
    parser.add_argument("--init-dataset-with-global-rank", action="store_true")
    parser.add_argument("--grid-run", action="store_true")
    parser.add_argument("--grid-run-frameworks", nargs="+", default=None)
    parser.add_argument(
        "--best-config",
        type=str,
        default=None,
        help="Only run experiments specified in the best config file",
    )
    parser.add_argument("--dcp-log-executor", action="store_true")
    parser.add_argument("--dcp-log-schedule", action="store_true")
    parser.add_argument(
        "--dry-run", action="store_true", help="Do not launch actual processes"
    )

    args = parser.parse_args()

    if args.grid_run_frameworks is not None:
        # check if the frameworks are valid
        for framework in args.grid_run_frameworks:
            assert framework in [
                "dcp",
                "mlm_a2a_p2p",
            ], f"Invalid framework {framework} for grid run"

    # init kvstore for exp control
    kvstore = RedisKVStore(
        args.host_ip, args.node_rank, args.num_nodes, args.framework
    )
    args.kvstore = kvstore
    return args


if __name__ == "__main__":
    args = _parse_args()
    run_grid_experiments(args)
    print("Experiment finished.")
