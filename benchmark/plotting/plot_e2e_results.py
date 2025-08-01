import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
import argparse

from utils import (
    ExperimentConfig,
    parse_throughput_results,
    get_dataset_abbreviation,
)


def rename_framework(row):
    if row["Framework"] == "mlm_a2a_p2p":
        return "MLM"
    elif row["Framework"] == "dcp":
        return "DCP"
    else:
        return row["Framework"]


def rename_mask(row):
    if row["MaskType"] == "causal":
        return "Causal"
    elif row["MaskType"] == "lambda":
        return "Lambda"
    elif row["MaskType"] == "causal_blockwise":
        return "Causal\nBlockwise"
    elif row["MaskType"] == "shared_question":
        return "Shared\nQuestion"
    else:
        return row["MaskType"]


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
            "MeanIterTime",
        ],
    )
    return df


def plot_e2e_results(exp_dir: str, out_dir: str):
    df = read_experiment_data(exp_dir)
    if df.empty:
        print("No data found in the specified directory.")
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
    # get the row with minimum MeanIterTime for each group
    min_dcp_indices = grouped_dcp_df["MeanIterTime"].idxmin()
    min_dcp_df = dcp_df.loc[min_dcp_indices].copy()

    # rename the frameworks and masks for better presentation
    min_dcp_df["Framework"] = min_dcp_df.apply(rename_framework, axis=1)
    min_dcp_df["Mask"] = min_dcp_df.apply(rename_mask, axis=1)
    mlm_df["Framework"] = mlm_df.apply(rename_framework, axis=1)
    mlm_df["Mask"] = mlm_df.apply(rename_mask, axis=1)

    for dataset in set(mlm_df["Dataset"].unique()).union(
        min_dcp_df["Dataset"].unique()
    ):
        for i, max_seq_len in enumerate(
            sorted(list(mlm_df["MaxSeqLen"].unique()))
        ):
            fig, ax = plt.subplots(1, 1, figsize=(5, 3))
            ax.set_xlabel("Mask", fontsize=12)
            ax.set_ylabel("Iteration Time (s)", fontsize=12)

            # Filter data for the current max_seq_len
            mlm_subset = mlm_df[
                (mlm_df["MaxSeqLen"] == max_seq_len)
                & (mlm_df["Dataset"] == dataset)
            ]
            dcp_subset = min_dcp_df[
                (min_dcp_df["MaxSeqLen"] == max_seq_len)
                & (min_dcp_df["Dataset"] == dataset)
            ]

            joined_df = pd.concat(
                [mlm_subset, dcp_subset], axis=0, ignore_index=True
            )

            joined_df["MeanIterTime"] = (
                joined_df["MeanIterTime"].astype(float) / 1000.0
            )

            # tick label formatter
            ax.yaxis.set_major_formatter(
                plt.FuncFormatter(lambda x, _: "{:.1f}".format(x))
            )

            joined_df["Mask"] = pd.Categorical(
                joined_df["Mask"],
                categories=[
                    "Causal",
                    "Lambda",
                    "Causal\nBlockwise",
                    "Shared\nQuestion",
                ],
                ordered=True,
            )

            sns.barplot(
                data=joined_df,
                x="Mask",
                y="MeanIterTime",
                hue="Framework",
                hue_order=["MLM", "DCP"],
                ax=ax,
                legend=False if i > 0 else True,
            )

            # set y axis limit
            if i < 2:
                ax.set_ylim(0, ax.get_ylim()[1] * 1.2)

            if i == 0:
                handles, labels = ax.get_legend_handles_labels()
                ax.legend(
                    handles, labels, ncols=2, loc="upper left", fontsize=12
                )

            # set x tick label font size
            ax.tick_params(axis="x", labelsize=12)
            # set y tick label font size
            ax.tick_params(axis="y", labelsize=12)

            # add grid lines
            ax.grid(axis="y", linestyle="--", alpha=0.5)

            filename = f"./e2e_{get_dataset_abbreviation(dataset)}_msl{max_seq_len}.pdf"
            fig.savefig(os.path.join(out_dir, filename), bbox_inches="tight")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot E2E results (Fig.14 & 15) from experiment directories."
    )
    parser.add_argument(
        "--exp-dir",
        type=str,
        required=True,
        help="Path to the experiment directory containing results.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="./",
        help="Directory to save the output plots.",
    )
    args = parser.parse_args()

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    plot_e2e_results(args.exp_dir, args.out_dir)
    print(f"Plots saved to {args.out_dir}.")
