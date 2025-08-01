from typing import Optional
import os
import argparse
import subprocess

MAX_SEQ_LEN = 65536


def run_cmd(
    num_nodes: int,
    n_gpus_per_node: int,
    hf_home: str,
    dataset: str,
    dataset_text_key: str,
    mask: str,
    n_iters: Optional[int] = 200,
    dcp_block_size: Optional[int] = 1024,
    mem_balance_factor: Optional[float] = None,
    comp_balance_factor: Optional[float] = None,
    internode_comp_balance_factor: Optional[float] = None,
    lambda_mask_l_pretrain: Optional[int] = None,
    shared_question_mask_n_answers: Optional[int] = None,
    causal_blockwise_mask_k_local: Optional[int] = None,
):
    """
    Run the grid search command.
    """
    cmd = f"""python3 benchmark/mlm/dry_run.py --hf-home {hf_home} \
        --num-nodes {num_nodes} \
        --num-gpus-per-node {n_gpus_per_node} \
        --dataset {dataset} \
        --dataset-text-key {dataset_text_key} \
        --mask-type {mask} \
        --n-iters {n_iters} \
        --max-seq-len {MAX_SEQ_LEN} \
        --n-tokens-per-global-batch {MAX_SEQ_LEN} \
        --dcp-block-size {dcp_block_size} \
        """
    if mem_balance_factor is not None:
        cmd += f"--dcp-mem-imbalance-epsilon {mem_balance_factor} "
    if comp_balance_factor is not None:
        cmd += f"--dcp-comp-imbalance-epsilon {comp_balance_factor} "
    if internode_comp_balance_factor is not None:
        cmd += f"--dcp-inter-node-comp-imbalance-factor {internode_comp_balance_factor} "
    if mask == "lambda" and lambda_mask_l_pretrain is not None:
        cmd += f"--lambda-mask-l-pretrain {lambda_mask_l_pretrain} "
    if (
        mask == "shared_question"
        and shared_question_mask_n_answers is not None
    ):
        cmd += f"--shared-question-mask-n-answers {shared_question_mask_n_answers} "
    if (
        mask == "causal_blockwise"
        and causal_blockwise_mask_k_local is not None
    ):
        cmd += (
            f"--causal-blockwise-mask-k-local {causal_blockwise_mask_k_local} "
        )

    p = subprocess.Popen(
        cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    success = False
    while p.poll() is None:
        output = p.stdout.readline()
        if output:
            print(output.decode().strip(), flush=True)
            if "Obtained all batches." in output.decode():
                success = True
                break
    if p.returncode != 0 and not success:
        print(f"Command failed with return code {p.returncode}", flush=True)
        stderr_output = p.stderr.read().decode()
        if stderr_output:
            print(f"Error: {stderr_output}", flush=True)
    p.kill()
    os.system("pkill -9 -f 'dry_run.py'")
    os.system("pkill -9 -f 'redis-server'")


def run_grid_search(dataset, key, num_nodes, num_gpus_per_node, hf_home):
    # first run causal mask on the dataset
    print(f"Running causal mask baseline exp for {dataset}...")
    run_cmd(
        num_nodes,
        num_gpus_per_node,
        hf_home,
        dataset,
        key,
        "causal",
    )
    # run the block size exps
    print(f"Running block size exps...", flush=True)
    BLOCK_SIZE = [512, 1024, 2048, 4096]
    for mask in [
        "causal",
        "lambda",
        "shared_question",
        "causal_blockwise",
    ]:
        for dcp_block_size in BLOCK_SIZE:
            print(
                f"Running {mask} mask with block size {dcp_block_size}...",
                flush=True,
            )
            run_cmd(
                num_nodes,
                num_gpus_per_node,
                hf_home,
                dataset,
                key,
                mask,
                dcp_block_size=dcp_block_size,
            )
    # then run the mask sparsity exps
    LAMBDA_MASK_L_PRETRAIN = [512, 1024, 2048, 4096, 8192]
    for l in LAMBDA_MASK_L_PRETRAIN:
        print(f"Running lambda mask with l_pretrain {l}...", flush=True)
        run_cmd(
            num_nodes,
            num_gpus_per_node,
            hf_home,
            dataset,
            key,
            "lambda",
            lambda_mask_l_pretrain=l,
        )
    # then run the shared question mask exps
    SHARED_QUESTION_MASK_N_ANSWERS = [1, 2, 4, 8, 16]
    for n in SHARED_QUESTION_MASK_N_ANSWERS:
        print(
            f"Running shared question mask with n_answers {n}...",
            flush=True,
        )
        run_cmd(
            num_nodes,
            num_gpus_per_node,
            hf_home,
            dataset,
            key,
            "shared_question",
            shared_question_mask_n_answers=n,
        )
    # then run the causal blockwise mask exps
    CAUSAL_BLOCKWISE_MASK_DCP_K_LOCAL = [1, 2, 4, 8, 16]
    for k in CAUSAL_BLOCKWISE_MASK_DCP_K_LOCAL:
        print(
            f"Running causal blockwise mask with k_local {k}...",
            flush=True,
        )
        run_cmd(
            num_nodes,
            num_gpus_per_node,
            hf_home,
            dataset,
            key,
            "causal_blockwise",
            causal_blockwise_mask_k_local=k,
        )
    # run the balance factor exps
    MEM_BALANCE_FACTORS = [0.1, 0.2, 0.3, 0.4]
    INTERNODE_COMP_BALANCE_SCALE_FACTORS = [1, 2, 4, 8, 16]
    for mem_balance_factor in MEM_BALANCE_FACTORS:
        for (
            internode_comp_balance_factor
        ) in INTERNODE_COMP_BALANCE_SCALE_FACTORS:
            print(
                f"Running causal mask with mem_balance_factor {mem_balance_factor} and internode_comp_balance_factor {internode_comp_balance_factor}...",
                flush=True,
            )
            run_cmd(
                num_nodes,
                num_gpus_per_node,
                hf_home,
                dataset,
                key,
                "causal",
                mem_balance_factor=mem_balance_factor,
                internode_comp_balance_factor=internode_comp_balance_factor,
            )


if __name__ == "__main__":
    # Set the HF_HOME environment variable to the path of your Hugging Face cache
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--hf-home",
        type=str,
        help="Path to the Hugging Face cache",
        default="~/.cache/huggingface",
    )
    parser.add_argument(
        "--num-nodes",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--n-gpus-per-node",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="LongAlign",
        choices=["LongAlign", "LDC"],
        help="Dataset to use",
    )
    args = parser.parse_args()

    if args.dataset == "LongAlign":
        dataset = "THUDM/LongAlign-10k"
        dataset_text_key = "messages"
    elif args.dataset == "LDC":
        dataset = "jchenyu/Long-Data-Collections-sample-10000"
        dataset_text_key = "text"
    else:
        raise ValueError(
            "Invalid dataset choice. Choose 'LongAlign' or 'LDC'."
        )
    run_grid_search(
        dataset,
        dataset_text_key,
        args.num_nodes,
        args.n_gpus_per_node,
        args.hf_home,
    )
