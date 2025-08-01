import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

DATASETS = ["LongAlign", "LongDataCollection"]
DATASET_NPY_FN = ["longalign_seqlens.npy", "ldc_seqlens.npy"]


def get_seqlens(dataset: str, dataset_npy_fn: str) -> np.ndarray:
    """
    Load the sequence lengths from a .npy file.
    """
    seqlens = np.load(
        f"{os.path.dirname(__file__)}/{dataset}/{dataset_npy_fn}"
    )
    return seqlens


def plot_seqlen_distribution(font_size: int = 16) -> None:
    """
    Plot the sequence length distribution for each dataset.
    """
    fig, ax = plt.subplots(figsize=(8, 3))
    for dataset, dataset_npy_fn in zip(DATASETS, DATASET_NPY_FN):
        seqlens = get_seqlens(dataset, dataset_npy_fn)
        sns.histplot(seqlens, bins=100, label=dataset, ax=ax, legend=True)

    l = ax.get_legend_handles_labels()
    ax.legend(l[0], l[1], loc="upper right", fontsize=font_size)

    plt.xlim(0, 131072)
    # plt.ticklabel_format(axis='y', style='sci', scilimits=(4,4))

    ax.tick_params(axis="x", labelsize=font_size)
    ax.tick_params(axis="y", labelsize=font_size)
    plt.xticks(fontsize=font_size)
    plt.yticks(fontsize=font_size)

    plt.title("Sequence Length Distribution", fontsize=font_size)
    plt.xlabel("Sequence Length", fontsize=font_size)
    plt.ylabel("Frequency", fontsize=font_size)
    plt.grid()
    plt.tight_layout()
    plt.savefig(
        f"{os.path.dirname(__file__)}/seqlen_distribution.pdf",
        bbox_inches="tight",
    )
    # plt.show()


if __name__ == "__main__":
    plot_seqlen_distribution(16)
