import os
from typing import List

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

FONT_SIZE = 14
LEGEND_FONT_SIZE = 12
BALANCE_FONT_SIZE = 11
MSL = 65536
DATASETS = [
    "THUDM_LongAlign-10k",
    "jchenyu_Long-Data-Collections-sample-10000",
]


def remap_mask_name(row):
    if row["mask"] == "lambda":
        return "Lambda"
    elif row["mask"] == "shared_question":
        return "Shared Question"
    elif row["mask"] == "causal_blockwise":
        return "Causal Blockwise"
    else:
        return "Causal"


def plot_block_size_vs_total_comm(
    block_size_exps_df: pd.DataFrame, dataset: str, out_dir: str
):
    fig, ax = plt.subplots(1, 1, figsize=(4.5, 2.5))

    ref_internode_cost = block_size_exps_df[
        block_size_exps_df["dataset"] == dataset
    ]["ref_internode_cost"].mean()

    # first plot the block size vs. inter-node cost
    sns.lineplot(
        data=block_size_exps_df[block_size_exps_df["dataset"] == dataset],
        x="Block Size",
        y="mean_internode_cost",
        hue="Mask",
        style="Mask",
        ax=ax,
        markers=True,
        dashes=False,
    )
    # ax.set_title(f"Total Inter-node Comm. Volume", fontsize=FONT_SIZE)
    ax.set_xlabel("Block Size", fontsize=FONT_SIZE)
    ax.set_ylabel("Comm Volume (MB)", fontsize=FONT_SIZE)

    # add a horizontal line for the reference inter-node cost
    ax.axhline(
        ref_internode_cost,
        color="black",
        linestyle="--",
        label="Reference Comm Volume",
    )

    # set ylim
    ax.set_ylim(ax.get_ylim()[0], ref_internode_cost * 1.1)
    # add text for the reference inter-node cost
    ax.text(
        ax.get_xlim()[0] + 0.1,
        ref_internode_cost - 10,
        "MLM Comm Volume",
        color="black",
        ha="left",
        va="top",
        fontsize=FONT_SIZE,
    )

    # set x ticks font size
    ax.tick_params(axis="x", labelsize=FONT_SIZE)
    # set y ticks font size
    ax.tick_params(axis="y", labelsize=FONT_SIZE)

    ax.grid(True, linestyle="--", alpha=0.5)

    # set y tick locator
    from matplotlib.ticker import MultipleLocator

    ax.yaxis.set_major_locator(MultipleLocator(100))

    # recalculate legend position
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        handles[:-1],
        labels[:-1],
        ncols=2,
        loc="upper left",
        fontsize=LEGEND_FONT_SIZE,
        title_fontsize=LEGEND_FONT_SIZE,
        bbox_to_anchor=(0.02, 0.75),
        borderaxespad=0,
        columnspacing=0.4,
    )
    filename = f"block_size_vs_comm_vol_{dataset}_msl{MSL}.pdf"
    fig.savefig(os.path.join(out_dir, filename), bbox_inches="tight")


def plot_block_size_vs_compilation_time(
    block_size_exps_df: pd.DataFrame, dataset: str, out_dir: str
):
    # then plot the block size vs. compilation time
    fig, ax = plt.subplots(1, 1, figsize=(4.5, 2.5))
    # ax = axes[2]
    sns.lineplot(
        data=block_size_exps_df[block_size_exps_df["dataset"] == dataset],
        x="Block Size",
        y="mean_compilation_time",
        hue="Mask",
        style="Mask",
        ax=ax,
        markers=True,
        dashes=False,
        legend=True if dataset == "THUDM_LongAlign-10k" else False,
    )
    # ax.set_title(f"Compilation Time Per Iteration", fontsize=FONT_SIZE)
    ax.set_xlabel("Block Size", fontsize=FONT_SIZE)
    ax.set_ylabel("Planning Time (s)", fontsize=FONT_SIZE)
    # set x ticks font size
    ax.tick_params(axis="x", labelsize=FONT_SIZE)
    # set y ticks font size
    ax.tick_params(axis="y", labelsize=FONT_SIZE)

    ax.grid(True, linestyle="--", alpha=0.5)
    # set y tick locator
    from matplotlib.ticker import MultipleLocator

    ax.yaxis.set_major_locator(
        MultipleLocator(10 if dataset == "THUDM_LongAlign-10k" else 5)
    )

    if dataset == "THUDM_LongAlign-10k":
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(
            handles,
            labels,
            ncols=1,
            loc="best",
            fontsize=LEGEND_FONT_SIZE,
            title_fontsize=LEGEND_FONT_SIZE,
            columnspacing=0.4,
        )
    filename = f"block_size_vs_compilation_time_{dataset}_msl{MSL}.pdf"
    fig.savefig(os.path.join(out_dir, filename), bbox_inches="tight")


def plot_mask_sparsity_vs_internode_comm_cost(
    causal_df: pd.DataFrame,
    mask_sparsity_exp_df: pd.DataFrame,
    dataset: str,
    out_dir: str,
):
    fig, ax = plt.subplots(1, 1, figsize=(5, 2))

    causal_internode_cost = causal_df[causal_df["dataset"] == dataset][
        "mean_internode_cost"
    ]

    # first plot the lambda mask
    sns.lineplot(
        data=mask_sparsity_exp_df[mask_sparsity_exp_df["dataset"] == dataset],
        x="mask_sparsity",
        y="mean_internode_cost",
        hue="Mask",
        ax=ax,
        style="Mask",
        markers=True,
        dashes=False,
        legend=True if dataset == "THUDM_LongAlign-10k" else False,
    )
    # ax.set_title(f"Lambda Sparsity vs. Inter-node Cost")
    ax.set_xlabel("Mask Sparsity", fontsize=12)
    ax.set_ylabel("Comm. Volume (MB)", fontsize=12)

    # adjust ylim
    ax.set_ylim(
        ax.get_ylim()[0],
        causal_internode_cost.mean()
        * (1.2 if dataset == "THUDM_LongAlign-10k" else 1.1),
    )

    # add a horizontal line for the reference inter-node cost
    ax.axhline(
        causal_internode_cost.mean(),
        color="black",
        linestyle="--",
        label="Causal Inter-node Cost",
    )
    # add text for the reference inter-node cost
    ax.text(
        ax.get_xlim()[0] + 0.05,
        causal_internode_cost.mean() + 0.1,
        "Causal Mask Comm. Volume",
        color="black",
        ha="left",
        va="bottom",
        fontsize=12,
    )
    # set x ticks font size
    ax.tick_params(axis="x", labelsize=FONT_SIZE)
    # set y ticks font size
    ax.tick_params(axis="y", labelsize=FONT_SIZE)
    ax.grid(True, linestyle="--", alpha=0.5)
    # recalculate legend position
    if dataset == "THUDM_LongAlign-10k":
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(
            handles[:-1],
            labels[:-1],
            ncol=1,
            loc="lower right",
            fontsize=10,
            title_fontsize=10,
            bbox_to_anchor=(0.98, 0.05),
            borderaxespad=0,
        )
    filename = f"mask_sparsity_vs_comm_vol_{dataset}_msl{MSL}.pdf"
    fig.savefig(os.path.join(out_dir, filename), bbox_inches="tight")


def plot_balance_factor_vs_internode_comm_cost(
    balance_factor_df: pd.DataFrame, out_dir: str
):
    fig, ax = plt.subplots(1, 1, figsize=(6, 1.5))

    sns.lineplot(
        data=balance_factor_df[(balance_factor_df["mem_imbal"] == 0.1)],
        x="mean_compute_imbalance",
        y="mean_internode_cost",
        hue="dataset",
        style="dataset",
        ax=ax,
        markers=True,
        legend=True,
    )
    # ax.set_title(f"Balance Factor vs. Inter-node Cost")
    ax.set_xlabel("Computation Imbalance", fontsize=BALANCE_FONT_SIZE)
    ax.set_ylabel("Comm. Vol. (MB)", fontsize=BALANCE_FONT_SIZE)
    # set x ticks font size
    ax.tick_params(axis="x", labelsize=BALANCE_FONT_SIZE)
    # set y ticks font size
    ax.tick_params(axis="y", labelsize=BALANCE_FONT_SIZE)

    ax.grid(True, linestyle="--", alpha=0.5)

    # set y tick locator
    from matplotlib.ticker import MultipleLocator

    ax.yaxis.set_major_locator(MultipleLocator(20))

    handle, labels = ax.get_legend_handles_labels()
    ax.legend(
        handle,
        ["LongAlign", "LongDataCollections"],
        ncol=1,
        loc="best",
        fontsize=BALANCE_FONT_SIZE,
    )
    filename = f"balance_factor_vs_comm_vol_msl{MSL}.pdf"
    fig.savefig(os.path.join(out_dir, filename), bbox_inches="tight")


def plot_dryrun_results(csv_files: List[str], out_dir: str):
    """
    Preprocess dryrun results and save to a CSV file.
    """
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # Initialize a list to hold all dataframes
    dataframes = []

    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        dataframes.append(df)

    # concatenate all dataframes into one
    df = pd.concat(dataframes, ignore_index=True)

    # block size exps
    default_imbal_df = df[
        (df["mem_imbal"] == 0.2)
        & (df["compute_imbal"] == 0.1)
        & (df["internode_comp_imbal_factor"] == 4.0)
    ].copy()
    block_size_exps_causal_df = default_imbal_df[
        (default_imbal_df["lambda_lpretrain"] == -1)
        & (default_imbal_df["shared_question_nans"] == -1)
        & (default_imbal_df["causal_blockwise_klocal"] == -1)
    ]
    block_size_exps_lambda_df = default_imbal_df[
        (default_imbal_df["mask"] == "lambda")
        & (default_imbal_df["lambda_lpretrain"] == 4096)
    ]
    block_size_exps_shared_question_df = default_imbal_df[
        (default_imbal_df["mask"] == "shared_question")
        & (default_imbal_df["shared_question_nans"] == 4)
    ]
    block_size_exps_causal_blockwise_df = default_imbal_df[
        (default_imbal_df["mask"] == "causal_blockwise")
        & (default_imbal_df["causal_blockwise_klocal"] == 2)
    ]

    block_size_exps_df = pd.concat(
        [
            block_size_exps_causal_df,
            block_size_exps_lambda_df,
            block_size_exps_shared_question_df,
            block_size_exps_causal_blockwise_df,
        ],
        axis=0,
    )

    block_size_exps_df["max_internode_cost_after_schedule"] = (
        block_size_exps_df["max_internode_cost_after_schedule"] / 1e6
    )

    block_size_exps_df["Block Size"] = pd.Categorical(
        block_size_exps_df["dcp_block_size"].astype(str),
        categories=[
            str(x)
            for x in sorted(
                list(block_size_exps_df["dcp_block_size"].unique())
            )
        ],
        ordered=True,
    )
    block_size_exps_df_all_msls = block_size_exps_df.copy()
    block_size_exps_df_all_msls["Mask"] = block_size_exps_df_all_msls.apply(
        remap_mask_name, axis=1
    )

    block_size_exps_df = block_size_exps_df_all_msls[
        block_size_exps_df_all_msls["max_seq_len"] == MSL
    ].copy()

    for dataset in DATASETS:
        plot_block_size_vs_total_comm(block_size_exps_df, dataset, out_dir)
        plot_block_size_vs_compilation_time(
            block_size_exps_df, dataset, out_dir
        )

    # mask sparsity exp dfs
    mask_sparsity_exp_df = default_imbal_df[
        default_imbal_df["dcp_block_size"] == 1024
    ]
    causal_df = mask_sparsity_exp_df[mask_sparsity_exp_df["mask"] == "causal"]
    lambda_df = mask_sparsity_exp_df[mask_sparsity_exp_df["mask"] == "lambda"]
    shared_question_df = mask_sparsity_exp_df[
        mask_sparsity_exp_df["mask"] == "shared_question"
    ]
    causal_blockwise_df = mask_sparsity_exp_df[
        mask_sparsity_exp_df["mask"] == "causal_blockwise"
    ]

    mask_sparsity_exp_df = pd.concat(
        [lambda_df, shared_question_df, causal_blockwise_df], axis=0
    )
    mask_sparsity_exp_df["Mask"] = mask_sparsity_exp_df.apply(
        remap_mask_name, axis=1
    )

    for dataset in DATASETS:
        plot_mask_sparsity_vs_internode_comm_cost(
            causal_df, mask_sparsity_exp_df, dataset, out_dir
        )

    balance_factor_df = df[
        (df["mask"] == "causal") & (df["dcp_block_size"] == 1024)
    ].copy()

    balance_factor_df["max_internode_cost_after_schedule"] = (
        balance_factor_df["max_internode_cost_after_schedule"] / 1e6
    )
    plot_balance_factor_vs_internode_comm_cost(balance_factor_df, out_dir)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Plot dryrun results")
    parser.add_argument(
        "--csv-files",
        type=str,
        nargs="+",
        required=True,
        help="List of CSV files to process",
    )
    parser.add_argument(
        "--out-dir", type=str, required=True, help="Output directory for plots"
    )

    args = parser.parse_args()

    if not args.csv_files:
        raise ValueError(
            "No CSV files provided. Please specify at least one CSV file."
        )
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    plot_dryrun_results(args.csv_files, args.out_dir)
    print(f"Plots saved to {args.out_dir}.")
