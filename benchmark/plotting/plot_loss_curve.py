from typing import List
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

from utils import (
    ExperimentConfig,
    parse_throughput_results,
    parse_loss_curve,
    get_dataset_abbreviation,
)

MASK_TYPES = ["causal", "lambda", "causal_blockwise", "shared_question"]
MASK_TYPE_PRINT_NAME = [
    "Causal",
    "Lambda",
    "Causal Blockwise",
    "Shared Question",
]


def read_experiment_data(exp_dir: str):
    data = []
    dataset_dirs = [os.path.join(exp_dir, d) for d in os.listdir(exp_dir)]
    dataset_dirs = [d for d in dataset_dirs if os.path.isdir(d)]
    for dataset_dir in dataset_dirs:
        dataset_name = os.path.basename(dataset_dir)
        framework_dirs = [
            os.path.join(dataset_dir, d) for d in os.listdir(dataset_dir)
        ]
        framework_dirs = [d for d in framework_dirs if os.path.isdir(d)]
        for framework_dir in framework_dirs:
            framework_name = os.path.basename(framework_dir)
            exp_dirs = [
                os.path.join(framework_dir, d)
                for d in os.listdir(framework_dir)
            ]
            exp_dirs = [d for d in exp_dirs if os.path.isdir(d)]
            for exp_dir in exp_dirs:
                exp_config = ExperimentConfig.parse_history_experiments(
                    exp_dir
                )
                if exp_config.status == "success":
                    per_iteration_losses = parse_loss_curve(exp_dir)
                    if not per_iteration_losses:
                        continue
                    iteration_times = parse_throughput_results(exp_dir)
                    if not iteration_times:
                        continue
                    mean_iter_times = np.mean(iteration_times)
                    data.append(
                        [
                            dataset_name,
                            framework_name,
                            exp_config.max_seq_len,
                            exp_config.n_tokens_per_global_batch,
                            exp_config.block_size,
                            exp_config.head_block_size,
                            exp_config.mask_type,
                            exp_config.tp_size,
                            exp_config.cp_size,
                            exp_config.dp_size,
                            exp_config.mem_imbalance_epsilon,
                            exp_config.comp_imbalance_epsilon,
                            exp_config.inter_node_comp_imbalance_factor,
                            mean_iter_times,
                            per_iteration_losses,
                        ]
                    )

    df = pd.DataFrame(
        data,
        columns=[
            "Dataset",
            "Framework",
            "MaxSeqLen",
            "GlobalBatchSize",
            "BlockSize",
            "HeadBlockSize",
            "MaskType",
            "TPSize",
            "CPSize",
            "DPSize",
            "MemImbalanceEpsilon",
            "CompImbalanceEpsilon",
            "InterNodeCompImbalanceFactor",
            "MeanIterationTime",
            "LossesPerIteration",
        ],
    )
    return df


def plot_loss_curve(
    exp_dir: str,
    out_dir: str,
    dataset: str = "THUDM_LongAlign-10k",
    max_seq_len: int = 131072,
):
    df = read_experiment_data(exp_dir)
    if df.empty:
        print("No valid experiment data found.")
        return

    mlm_df = df[df["Framework"] == "mlm_a2a_p2p"].copy()
    dcp_df = df[df["Framework"] == "dcp"].copy()

    grouped_dcp_df = dcp_df.groupby(
        [
            "Dataset",
            "Framework",
            "MaxSeqLen",
            "GlobalBatchSize",
            "MaskType",
            "TPSize",
            "CPSize",
            "DPSize",
        ]
    )
    # get the row with minimum MeanIterationTime for each group
    min_dcp_indices = grouped_dcp_df["MeanIterationTime"].idxmin()

    min_dcp_df = dcp_df.loc[min_dcp_indices].copy()

    min_df = pd.concat([mlm_df, min_dcp_df], ignore_index=True)

    min_dataset_df = min_df[min_df["Dataset"] == dataset].copy()
    if min_dataset_df.empty:
        print(f"No data found for dataset {dataset}.")
        return

    for mask_type, mask_name in zip(MASK_TYPES, MASK_TYPE_PRINT_NAME):
        fig, ax = plt.subplots(1, 1, figsize=(4, 2))
        ax: List[plt.Axes]
        mask_df = min_dataset_df[min_dataset_df["MaxSeqLen"] == max_seq_len]
        mask_df = mask_df[mask_df["MaskType"] == mask_type]
        losses_dcp = mask_df[mask_df["Framework"] == "dcp"][
            "LossesPerIteration"
        ].values[0]
        losses_mlm = mask_df[mask_df["Framework"] == "mlm_a2a_p2p"][
            "LossesPerIteration"
        ].values[0]
        losses = losses_mlm + losses_dcp
        labels = ["MLM"] * len(losses_mlm) + ["DCP"] * len(losses_dcp)
        iterations = list(range(len(losses_mlm))) + list(
            range(len(losses_dcp))
        )
        df = pd.DataFrame(
            {"Iteration": iterations, "Loss": losses, "Framework": labels}
        )
        sns.lineplot(
            data=df,
            x="Iteration",
            y="Loss",
            hue="Framework",
            ax=ax,
            style="Framework",
            dashes=[(1, 0), (2, 1)],
        )
        ax.set_title(f"Mask: {mask_name}", fontsize=12)
        ax.set_xlabel("Iteration", fontsize=12)
        ax.set_ylabel("Loss", fontsize=12)
        ax.tick_params(axis="both", labelsize=12)
        # ax.set_xticklabels(ax.get_xticklabels(), fontsize=12)
        # ax.set_yticklabels(ax.get_yticklabels(), fontsize=12)
        ax.grid()
        filename = f"loss_curve_{get_dataset_abbreviation(dataset)}_msl{max_seq_len}_mask{mask_type}.pdf"
        fig.savefig(os.path.join(out_dir, filename), bbox_inches="tight")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Plot loss curves from experiment data."
    )
    parser.add_argument(
        "--exp-dir",
        type=str,
        required=True,
        help="Path to the experiment directory.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=".",
        help="Output directory for the plots.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="LongAlign",
        choices=["LongAlign", "LDC"],
        help="Dataset to plot.",
    )
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=131072,
        help="Maximum sequence length plot.",
    )

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

    plot_loss_curve(args.exp_dir, args.out_dir, args.dataset, args.max_seq_len)
