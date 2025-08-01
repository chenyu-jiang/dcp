import os
from functools import partial

import argparse

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


def get_exp_spec_name(args):
    nranks = args.num_nodes * args.num_gpus_per_node
    dp_size = nranks // (args.tp_size * args.cp_size)
    exp_spec_name = "tp{}_dp{}_cp{}".format(
        args.tp_size, dp_size, args.cp_size
    )
    if args.dcp_use_block_size_heuristic:
        exp_spec_name += "_bsz-1"
    else:
        exp_spec_name += "_bsz{}".format(args.dcp_block_size)
    exp_spec_name += "_hbs{}".format(args.dcp_head_block_size)
    exp_spec_name += "_msl{}_gbs{}".format(
        args.max_seq_len, args.n_tokens_per_global_batch
    )
    balance_vals = "{:.1f},{:.1f},{:.1f}".format(
        args.dcp_mem_imbalance_epsilon,
        args.dcp_comp_imbalance_epsilon,
        args.dcp_inter_node_comp_imbalance_factor,
    )
    exp_spec_name += "_bal" + balance_vals
    exp_spec_name += "_msk{}".format(args.mask_type)
    if args.mask_type == "lambda":
        exp_spec_name += "_lpretrain{}".format(args.lambda_mask_l_pretrain)
    elif args.mask_type == "shared_question":
        exp_spec_name += "_nans{}".format(args.shared_question_mask_n_answers)
    elif args.mask_type == "causal_blockwise":
        exp_spec_name += "_klocal{}".format(args.causal_blockwise_mask_k_local)
    return exp_spec_name


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1234, help="Random seed")
    parser.add_argument(
        "--vocab_file",
        type=str,
        help="Path to the vocab file",
        default="./vocabs/gpt2-vocab.json",
    )
    parser.add_argument(
        "--merge_file",
        type=str,
        help="Path to the merge file",
        default="./vocabs/gpt2-merges.txt",
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
        "--hf-home",
        type=str,
        default="~/.cache/huggingface",
        help="The home directory of Hugging Face",
    )
    parser.add_argument("--tp-size", type=int, default=4)
    parser.add_argument("--cp-size", type=int, default=16)
    parser.add_argument("--max-seq-len", type=int, default=16384)
    parser.add_argument("--n-tokens-per-global-batch", type=int, default=32768)
    parser.add_argument("--n-iters", type=int, default=200)
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
        choices=["causal", "lambda", "shared_question", "causal_blockwise"],
    )
    parser.add_argument(
        "--lambda-mask-l-pretrain",
        type=int,
        default=4096,
    )
    parser.add_argument(
        "--shared-question-mask-n-answers",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--causal-blockwise-mask-k-local",
        type=int,
        default=2,
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
    parser.add_argument("--dcp-intra-node-lat", type=float, default=0.0)
    parser.add_argument("--dcp-intra-node-bw", type=float, default=4800)
    parser.add_argument("--dcp-inter-node-lat", type=float, default=0.0)
    parser.add_argument("--dcp-inter-node-bw", type=float, default=400)
    parser.add_argument("--dcp-max-tflops", type=float, default=280)
    parser.add_argument("--dcp-max-mem-bw", type=float, default=1500)
    # data loader and planner
    parser.add_argument(
        "--dcp-prefetch-planner-num-workers", type=int, default=64
    )
    parser.add_argument(
        "--dcp-prefetch-listener-num-workers", type=int, default=2
    )
    parser.add_argument("--dcp-block-size", type=int, default=1024)
    parser.add_argument(
        "--dcp-use-block-size-heuristic",
        action="store_true",
        help="Use block size heuristic",
    )
    args = parser.parse_args()
    args.dataset = [args.dataset]
    model_config = MODEL_CONFIGS[args.model]
    args.kv_channels = model_config["model_head_size"]
    args.num_attention_heads = model_config["model_n_heads"]
    args.num_query_groups = model_config["model_n_query_groups"]
    args.group_query_attention = (
        args.num_query_groups < args.num_attention_heads
    )
    args.bf16 = True
    args.pipeline_model_parallel_size = 1
    args.exp_spec_name = get_exp_spec_name(args)
    exp_logging_dir = os.path.join(
        "./dryrun_experiments",
        args.dataset[0].replace("/", "_"),
        args.exp_spec_name,
    )
    if not os.path.exists(exp_logging_dir):
        os.makedirs(exp_logging_dir)
    args.exp_logging_dir = os.path.join(exp_logging_dir)
    return args


args = parse_args()


os.environ["DCP_DEBUG"] = "DEBUG"
os.environ["DCP_LOG_GRAPH_PARTITION"] = "1"
os.environ["HF_HOME"] = args.hf_home
os.environ["DCP_LOG_INSTRS"] = "1"
os.environ["DCP_LOGGING_DEBUG_DIR"] = args.exp_logging_dir

from dcp.data.dataloader import DCPDataLoader, TrainingSpec
from dcp.core.instructions import DType
from dcp.core.common import ExecutionContext, ModelSpec
from dcp.core.cost_model import (
    CommunicationCostModel,
    AttnRooflineCostModel,
)
from dcp.data.hf_dataset import (
    HuggingFaceDataset,
    HuggingFaceDatasetConfig,
    OrderedBatchSampler,
)
from dcp.runtime.flash_attention.utils import (
    lambda_mask_fn,
    shared_question_mask_fn,
    causal_blockwise_mask_fn,
)

# requires Megatron-LM tokenizer
from megatron.training.tokenizer.tokenizer import _GPT2BPETokenizer

from run_experiments import MODEL_CONFIGS

_MASK_FNS = {
    "causal": None,
    "lambda": lambda_mask_fn,
    "shared_question": shared_question_mask_fn,
    "causal_blockwise": causal_blockwise_mask_fn,
}


def _get_mask_fn(args):
    if args.mask_type not in _MASK_FNS:
        raise ValueError(f"Unknown mask type: {args.mask_type}")
    if args.mask_type == "causal":
        return None
    if args.mask_type == "lambda":
        return partial(lambda_mask_fn, l_pretrain=args.lambda_mask_l_pretrain)
    if args.mask_type == "shared_question":
        return partial(
            shared_question_mask_fn,
            n_answers=args.shared_question_mask_n_answers,
        )
    if args.mask_type == "causal_blockwise":
        return partial(
            causal_blockwise_mask_fn,
            k_local=args.causal_blockwise_mask_k_local,
        )
    raise ValueError(f"Unknown mask type: {args.mask_type}. ")


def core_hf_dataset_config_from_args(args):
    tokenizer = _GPT2BPETokenizer(args.vocab_file, args.merge_file)
    assert len(args.dataset) == 1

    return HuggingFaceDatasetConfig(
        random_seed=args.seed,
        sequence_length=args.max_seq_len,
        data_path=args.dataset,
        split=args.dataset_split,
        tokenizer=tokenizer,
        num_preprocessing_workers=args.dcp_prefetch_planner_num_workers,
    )


def get_dataloader(args):
    config = core_hf_dataset_config_from_args(args)

    hf_dataset = config.data_path[0]

    train_ds = HuggingFaceDataset(
        hf_dataset,
        dataset_name=config.data_path[0],
        num_samples=None,
        index_split=args.dataset_split,
        text_key=args.dataset_text_key,
        config=config,
        max_seq_length=args.max_seq_len,
    )

    n_devices_per_node = args.num_gpus_per_node // args.tp_size
    n_nodes = args.num_nodes
    comm_cost_model = CommunicationCostModel(
        args.dcp_inter_node_lat,
        args.dcp_inter_node_bw,
        args.dcp_intra_node_lat,
        args.dcp_intra_node_bw,
    )
    comp_cost_model = AttnRooflineCostModel(
        args.dcp_max_tflops, args.dcp_max_mem_bw
    )
    buffer_size = args.dcp_prefetch_planner_num_workers
    listener_workers = args.dcp_prefetch_listener_num_workers
    dcp_size = args.cp_size
    tp_size = args.tp_size

    node_local_rank = 0

    exec_context = ExecutionContext(
        n_devices_per_node, n_nodes, comm_cost_model, comp_cost_model
    )

    training_spec = TrainingSpec(
        exec_context,
        ModelSpec(
            args.kv_channels,
            args.num_attention_heads // tp_size,
            (
                args.num_query_groups
                if args.group_query_attention
                else args.num_attention_heads
            )
            // tp_size,
        ),
        dcp_size=dcp_size,
        mask_type=args.mask_type,
        mask_dtype=DType.INT,
        qkv_dtype=(
            DType.BFLOAT16 if args.bf16 else DType.FLOAT16
        ),  # Checkout the dtype
        block_size=args.dcp_block_size,
        head_block_size=args.dcp_head_block_size,
        tensor_parallel_size=args.tp_size,
        pipeline_parallel_size=args.pipeline_model_parallel_size,
        use_block_size_heuristic=args.dcp_use_block_size_heuristic,
        mem_imbalance_epsilon=args.dcp_mem_imbalance_epsilon,
        comp_imbalance_epsilon=args.dcp_comp_imbalance_epsilon,
        inter_node_comp_imbalance_factor=args.dcp_inter_node_comp_imbalance_factor,
    )
    dataloader = DCPDataLoader(
        training_spec,
        train_ds,
        is_kv_host=True,
        node_rank=0,
        node_local_rank=node_local_rank,
        node_size=1,  # tricks the DataLoader to compute every iter locally
        dcp_rank=0,
        pp_rank=0,
        tp_rank=0,
        start_poller=True,
        batch_sampler=OrderedBatchSampler(
            train_ds, args.n_tokens_per_global_batch
        ),
        num_workers=listener_workers,
        num_preprocess_workers=buffer_size,
        pin_memory=False,
        input_key="tokens",
        mask_fn=_get_mask_fn(args),
        is_dryrun=True,
    )
    return dataloader


def dry_run(args):
    dataloader = get_dataloader(args)
    for i, batch in enumerate(dataloader):
        print("Obtained batch {}.".format(i), flush=True)
        if i == args.n_iters:
            break
    print("Obtained all batches.", flush=True)
    return


if __name__ == "__main__":
    dry_run(args)
