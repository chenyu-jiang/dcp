import os
from functools import partial

import argparse


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
        "--hf-home",
        type=str,
        default="~/.cache/huggingface",
        help="The home directory of Hugging Face",
    )
    parser.add_argument("--max-seq-len", type=int, default=16384)
    parser.add_argument("--dataset", type=str, default="THUDM/LongAlign-10k")
    parser.add_argument("--dataset-split", type=str, default="train")
    parser.add_argument("--dataset-text-key", type=str, default="messages")
    args = parser.parse_args()
    args.dataset = [args.dataset]
    return args


args = parse_args()


os.environ["HF_HOME"] = args.hf_home

from dcp.data.hf_dataset import (
    HuggingFaceDataset,
    HuggingFaceDatasetConfig,
)

# requires Megatron-LM tokenizer
from megatron.training.tokenizer.tokenizer import _GPT2BPETokenizer


def core_hf_dataset_config_from_args(args):
    tokenizer = _GPT2BPETokenizer(args.vocab_file, args.merge_file)
    assert len(args.dataset) == 1

    return HuggingFaceDatasetConfig(
        random_seed=args.seed,
        sequence_length=args.max_seq_len,
        data_path=args.dataset,
        split=args.dataset_split,
        tokenizer=tokenizer,
        num_preprocessing_workers=2,
    )


def get_dataset(args):
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

    return train_ds


def dry_run(args):
    train_ds = get_dataset(args)

    for i in range(20):
        print(train_ds[i]["tokens"].shape)


if __name__ == "__main__":
    dry_run(args)
