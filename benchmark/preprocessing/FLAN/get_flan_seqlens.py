import os
import sys

os.environ['HF_HOME'] = '/nfs/hf_cache'

import datasets
from datasets import load_dataset
import huggingface_hub
import numpy as np

import tiktoken

OUT_DIR = "./"

# hf_access_token = sys.argv[1]
# huggingface_hub.login(token=hf_access_token)

dataset = load_dataset(
    "Muennighoff/flan", split="train", num_proc=64, verification_mode=datasets.VerificationMode.NO_CHECKS
)
# shuffle and get the first 1M samples
dataset = dataset.shuffle(seed=42).select(range(1000000))
encoding = tiktoken.get_encoding("r50k_base")


def get_seqlen(sample):
    inputs = sample["inputs"]
    targets = sample["targets"]
    text = inputs + targets if targets else inputs
    tokenized = encoding.encode(text, allowed_special="all")
    sample["seqlen"] = len(tokenized)
    return sample


ds = dataset.map(get_seqlen, num_proc=64)

seqlens = np.array([x["seqlen"] for x in ds])

np.save(os.path.join(OUT_DIR, "flan_seqlens.npy"), seqlens)
