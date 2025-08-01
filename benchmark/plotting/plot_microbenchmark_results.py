import os
import argparse
from collections import defaultdict

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

HATCHES = ["//", "xx", "--", "..", "\\"]
HUE_ORDER = ["RFA (Ring)", "RFA (ZigZag)", "LT", "TE", "DCP"]
HATCHES_MASKS = ["..", "xx", "//", "..", "\\"]
HUE_ORDER_MASKS = ["TE FW", "DCP FW", "TE BW", "DCP BW"]
MASK_ORDER = ["Causal", "Causal\nBlockwise", "Lambda", "Shared\nQuestion"]

MEANS = [0.5, 1, 2, 4]

############################################################################
# Utility functions
############################################################################


def parse_dataset_name(dir_path: str):
    basename = os.path.basename(dir_path)
    parts = basename.split("_")
    assert (
        len(parts) == 4
    ), f"Failed to parse dataset name {basename} in {dir_path}."
    dataset = parts[0]
    max_seq_len = int(parts[1])
    mean = float(parts[2].replace("mean", ""))
    n_nodes = int(parts[3].split("N")[0])
    n_total_devices = int(parts[3].split("N")[1].split("D")[0])
    return dataset, max_seq_len, mean, n_nodes, n_total_devices


def load_dataframe_from_csv(dir_path: str):
    dataset, max_seq_len, mean, n_nodes, n_total_devices = parse_dataset_name(
        dir_path
    )
    fn = os.path.join(dir_path, "results.csv")
    if os.path.exists(fn):
        df = pd.read_csv(fn)
        df["mean"] = mean
    else:
        df = None
    return df, dataset, max_seq_len, mean, n_nodes, n_total_devices


def construct_dataframes(result_dir: str):
    exp_dirs = [
        d
        for d in os.listdir(result_dir)
        if os.path.isdir(os.path.join(result_dir, d))
    ]
    exp_dirs = [os.path.abspath(os.path.join(result_dir, d)) for d in exp_dirs]
    dfs = defaultdict(list)
    for dir_path in exp_dirs:
        df, dataset, max_seq_len, mean, n_nodes, n_total_devices = (
            load_dataframe_from_csv(dir_path)
        )
        if df is not None:
            df["mean"] = mean
            dfs[(dataset, max_seq_len, n_nodes, n_total_devices)].append(df)
    return dfs


def _rename_frameworks(row):
    if row["Framework"] == "dcp":
        return "DCP"
    elif row["Framework"] == "lt":
        return "LT"
    elif row["Framework"] == "te":
        return "TE"
    elif row["Framework"] == "ring":
        return "RFA (Ring)"
    elif row["Framework"] == "zigzag":
        return "RFA (ZigZag)"
    else:
        raise ValueError(f"Unknown framework: {row['Framework']}")


def _rename_fw_bw(row):
    if row["benchmark_type"] == "forward":
        return "FW"
    else:
        return "BW"


def _rename_mask_type(row):
    if row["mask_type"] == "causal":
        return "Causal"
    elif row["mask_type"] == "causal_blockwise":
        return "Causal\nBlockwise"
    elif row["mask_type"] == "lambda":
        return "Lambda"
    elif row["mask_type"] == "shared_question":
        return "Shared\nQuestion"
    else:
        raise ValueError(f"Unknown mask type: {row['mask_type']}")


############################################################################
# Plotting functions
############################################################################


# function to draw a cross in the bar plot for missing data (e.g., OOM)
def add_cross_at(
    ax: plt.Axes,
    x_index,
    hue,
    hue_order,
    size=80,
    y_offset=500,
):
    # x_index is the index of the bar (excluding hue)
    hue_index = hue_order.index(hue)
    hue_offset = (-(len(hue_order) / 2) + 0.5 + hue_index) * ax.patches[
        0
    ].get_width()
    ax.scatter(x_index + hue_offset, y_offset, marker="x", color="red", s=size)


# plot the results for causal masks
def plot_causal(
    output_dir: str,
    dataset: str,
    max_seq_len: int,
    n_nodes: int,
    n_total_devices: int,
    df: pd.DataFrame,
    is_forward: bool,
    legend=True,
    font_size=15,
    legend_fontsize=12,
):

    causal_df = df[
        (df["mask_type"] == "causal")
        & (df["Operation"] == ("FW" if is_forward else "BW"))
    ].copy()

    base = 10 if is_forward else 20
    loc = ticker.MultipleLocator(base=base)

    fig, ax = plt.subplots(1, 1, figsize=(8, 3))

    ax.grid(lw=0.4, ls="--", color="gray")

    sns.barplot(
        x="mean",
        y="avg_time",
        hue="Framework",
        data=causal_df,
        ax=ax,
        hue_order=HUE_ORDER,
        legend=legend,
    )
    ax.set_ylim(0, causal_df["avg_time"].max() * 1.03)
    ax.yaxis.set_major_locator(loc)
    ax.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.0f}"))

    # draw crosses for missing data
    y_offset = ax.get_ylim()[1] * 0.03
    size = 60
    for mean in MEANS:
        x_index = MEANS.index(mean)
        for hue in HUE_ORDER:
            if (
                len(
                    causal_df[
                        (causal_df["mean"] == mean)
                        & (causal_df["Framework"] == hue)
                    ]
                )
                == 0
            ):
                add_cross_at(
                    ax,
                    x_index,
                    hue,
                    hue_order=HUE_ORDER,
                    y_offset=y_offset,
                    size=size,
                )

    unique_patch_colors = []
    for i, bar in enumerate(ax.patches):
        if bar.get_facecolor() not in unique_patch_colors:
            unique_patch_colors.append(bar.get_facecolor())

    plt.rcParams["hatch.linewidth"] = 0.3
    for i, bar in enumerate(ax.patches):
        color_idx = unique_patch_colors.index(bar.get_facecolor())
        hatch = HATCHES[color_idx]
        bar.set_hatch(hatch)
        bar.set_edgecolor("white")

    l = ax.legend(
        loc="lower center",
        ncols=5,
        bbox_to_anchor=(0.5, 1),
        prop={"size": legend_fontsize},
        columnspacing=0.2,
        handletextpad=0.4,
    )
    ax.set_ylabel("Avg. Time (ms)", fontsize=font_size)
    ax.set_xlabel("Sequence Length Scale", fontsize=font_size)
    ax.tick_params(axis="both", labelsize=font_size)
    plt.tight_layout()
    filename = f"microbenchmark_{dataset}_msl{max_seq_len}_N{n_nodes}D{n_total_devices}_causal_{'FW' if is_forward else 'BW'}.pdf"
    plt.savefig(os.path.join(output_dir, filename), bbox_inches="tight")


def plot_masks(
    output_dir: str,
    dataset: str,
    max_seq_len: int,
    n_nodes: int,
    n_total_devices: int,
    df: pd.DataFrame,
    legend=True,
    x_label_font_size=14,
    font_size=16,
    legend_fontsize=14,
):

    framework_df = df[
        (df["Framework"] == "TE") | (df["Framework"] == "DCP")
    ].copy()
    mask_df = framework_df.copy()

    mask_df["mean_and_mask"] = (
        mask_df["mean"].astype(str) + " " + mask_df["Mask Type"]
    )
    mask_df["Framework_and_Op"] = (
        mask_df["Framework"] + " " + mask_df["Operation"]
    )

    ordered_mean_and_masks = sorted(
        list(mask_df["mean_and_mask"].unique()),
        key=lambda x: (
            float(x.split(" ")[0]),
            MASK_ORDER.index(x.split(" ")[1]),
        ),
    )

    fig, ax = plt.subplots(1, 1, figsize=(16, 4))

    ax.grid(lw=0.4, ls="--", color="gray")

    sns.barplot(
        x="mean_and_mask",
        y="avg_time",
        hue="Framework_and_Op",
        data=mask_df,
        ax=ax,
        hue_order=HUE_ORDER_MASKS,
        legend=legend,
        order=ordered_mean_and_masks,
    )
    ax.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.0f}"))

    xlabels = [l.get_text() for l in ax.get_xticklabels()]
    mask_labels = [x.split(" ")[1] for x in xlabels]
    mean_labels = [x.split(" ")[0] for x in xlabels]
    ax.set_xticks(ax.get_xticks())
    ax.set_xticklabels(mask_labels, size=14)
    ax.set_xlim(-0.5, len(MEANS) * len(MASK_ORDER) - 0.5)

    unique_patch_colors = []
    for i, bar in enumerate(ax.patches):
        if bar.get_facecolor() not in unique_patch_colors:
            unique_patch_colors.append(bar.get_facecolor())

    plt.rcParams["hatch.linewidth"] = 0.3
    for i, bar in enumerate(ax.patches):
        bar_idx = i // 16
        hatch_idx = bar_idx // 2
        color_idx = bar_idx % 2
        hatch = HATCHES_MASKS[hatch_idx]
        color = unique_patch_colors[color_idx]
        if hatch_idx == 0:
            color = (color[0] * 0.9, color[1] * 0.9, color[2] * 0.9, 1)
        else:
            color = (color[0] * 1.1, color[1] * 1.1, color[2] * 1.1, 1)
        bar.set_facecolor(color)
        bar.set_hatch(hatch)
        bar.set_edgecolor("white")

    l = ax.legend(
        loc="upper right",
        ncols=5,
        bbox_to_anchor=(1, 1),
        prop={"size": legend_fontsize},
        columnspacing=0.2,
        handletextpad=0.4,
    )
    ax.set_ylabel("Avg. Time (ms)", fontsize=font_size)
    ax.tick_params(axis="y", labelsize=font_size)
    ax.tick_params(axis="x", labelsize=x_label_font_size)

    # set legend color
    handles, labels = ax.get_legend_handles_labels()
    for i, handle in enumerate(handles):
        color_idx = i % 2
        hatch_idx = i // 2
        color = unique_patch_colors[color_idx]
        hatch = HATCHES_MASKS[hatch_idx]
        if hatch_idx == 0:
            color = (color[0] * 0.9, color[1] * 0.9, color[2] * 0.9, 1)
        else:
            color = (color[0] * 1.1, color[1] * 1.1, color[2] * 1.1, 1)
        handle.set_facecolor(color)
        handle.set_hatch(hatch)
        handle.set_edgecolor("white")
    ax.legend(
        handles,
        labels,
        loc="upper right",
        ncols=5,
        bbox_to_anchor=(1, 1),
        prop={"size": legend_fontsize},
        columnspacing=0.2,
        handletextpad=0.4,
        borderpad=0.2,
    )
    # get delimiter positions
    delimiter_positions = []
    for i, label in enumerate(mean_labels):
        if label != mean_labels[i - 1]:
            delimiter_positions.append(i)
    delimiter_positions.append(len(mean_labels))
    # add delimiters
    for pos in delimiter_positions:
        ax.axvline(
            x=pos - 0.5,
            ymin=-0.4,
            ymax=1,
            color="black",
            linestyle="--",
            lw=1,
            clip_on=False,
        )
    for mean, pos0, pos1 in zip(
        MEANS, delimiter_positions[:-1], delimiter_positions[1:]
    ):
        ax.text(
            (pos0 + pos1 - 1) / 2,
            -0.35,
            f"Mean Scale={mean}",
            ha="center",
            clip_on=False,
            transform=ax.get_xaxis_transform(),
            fontsize=font_size,
        )

    ax.set_xlabel("Mean Scale & Mask", fontsize=font_size)
    ax.xaxis.set_label_coords(0.5, -0.45)
    plt.tight_layout()
    filename = f"microbenchmark_{dataset}_msl{max_seq_len}_N{n_nodes}D{n_total_devices}_masks.pdf"
    plt.savefig(os.path.join(output_dir, filename), bbox_inches="tight")


def plot_results(exp_dir: str, out_dir: str):
    dataframes = construct_dataframes(exp_dir)
    assert len(dataframes) > 0, f"No experiment data found in {exp_dir}"
    for (
        dataset,
        max_seq_len,
        n_nodes,
        n_total_devices,
    ), df_list in dataframes.items():
        # merge the dataframes
        df = pd.concat(df_list)
        df = df.reset_index()
        df.drop(columns=["index"], inplace=True)
        # calculate average time
        df["avg_time"] = df["avg_time"] / df["n_iters"]
        df = df.dropna()
        # for lt baseline, with other params fixed, group by lt_window_size
        # and get minimum avg_time.
        # since other frameworks have only one lt_window_size
        # the result is not affected
        lt_groupby_cols = df.columns.tolist()
        lt_groupby_cols.remove("avg_time")
        lt_groupby_cols.remove("lt_window_size")
        lt_groupby_cols.remove("n_iters")
        lt_grouped = df.groupby(lt_groupby_cols)
        min_lt_df = df.loc[lt_grouped["avg_time"].idxmin()]
        # similarly for dcp, with other params fixed,
        # group by dcp_block_size and get minimum
        dcp_groupby_cols = df.columns.tolist()
        dcp_groupby_cols.remove("avg_time")
        dcp_groupby_cols.remove("dcp_block_size")
        dcp_groupby_cols.remove("n_iters")
        dcp_grouped = min_lt_df.groupby(dcp_groupby_cols)
        min_dcp_df = min_lt_df.loc[dcp_grouped["avg_time"].idxmin()]
        min_df = min_dcp_df.dropna()
        # rename framework for better presentation
        min_df = min_df.rename(columns={"framework": "Framework"})
        min_df["Framework"] = min_df.apply(_rename_frameworks, axis=1)
        # rename FW BW operation name
        min_df["Operation"] = min_df.apply(_rename_fw_bw, axis=1)
        min_df["Mask Type"] = min_df.apply(_rename_mask_type, axis=1)
        plot_causal(
            out_dir,
            dataset,
            max_seq_len,
            n_nodes,
            n_total_devices,
            min_df,
            is_forward=True,
        )
        plot_causal(
            out_dir,
            dataset,
            max_seq_len,
            n_nodes,
            n_total_devices,
            min_df,
            is_forward=False,
        )
        plot_masks(
            out_dir, dataset, max_seq_len, n_nodes, n_total_devices, min_df
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate Fig.12 and Fig.13 (microbenchmark results) in the paper."
    )
    parser.add_argument(
        "--exp-dir",
        type=str,
        required=True,
        help="Path to the directory containing the microbenchmark results.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="./",
        help="Path to the directory where the plots will be saved.",
    )
    args = parser.parse_args()
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)
    plot_results(args.exp_dir, args.out_dir)
    print(f"Plots saved to {args.out_dir}.")
