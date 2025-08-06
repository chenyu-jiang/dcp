import argparse
import json

from benchmark.plotting.plot_e2e_results import read_experiment_data


def get_best_configs(exp_dir: str) -> dict:
    df = read_experiment_data(exp_dir)
    if df.empty:
        print("No data found in the specified directory.")
        return

    dcp_df = df[df["Framework"] == "bblock"].copy()

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
    # drop the MeanIterTime and Framework columns
    min_dcp_df.drop(columns=["MeanIterTime", "Framework"], inplace=True)
    # convert to dictionary
    best_configs = min_dcp_df.to_dict(orient="records")
    return best_configs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Get best configurations from experiments."
    )
    parser.add_argument(
        "--exp-dir",
        type=str,
        required=True,
        help="Path to the experiment directory containing results.",
    )
    parser.add_argument(
        "--out-path",
        type=str,
        default="./best_configs.json",
        help="Output directory to save the best configurations.",
    )
    args = parser.parse_args()

    best_configs = get_best_configs(args.exp_dir)
    if best_configs:
        # create parent directories if they do not exist
        import os

        os.makedirs(os.path.dirname(args.out_path), exist_ok=True)
        # dump the best configurations to a JSON file
        with open(args.out_path, "w") as f:
            json.dump(best_configs, f, indent=4)
        print(f"Best configurations dumped to {args.out_path}")
    else:
        print("No best configurations found.")
