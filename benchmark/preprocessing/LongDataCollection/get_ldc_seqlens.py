import os
import sys

# os.environ['HF_HOME'] = '/opt/dlami/nvme/huggingface'

from datasets import load_dataset
import huggingface_hub
import numpy as np

import tiktoken

OUT_DIR = "./"

# hf_access_token = sys.argv[1]
# huggingface_hub.login(token=hf_access_token)

dataset = load_dataset(
    "jchenyu/Long-Data-Collections-sample-10000", split="train", num_proc=64
)
encoding = tiktoken.get_encoding("r50k_base")


def get_seqlen(sample):
    text = sample["text"]
    tokenized = encoding.encode(text, allowed_special="all")
    sample["seqlen"] = len(tokenized)
    return sample


ds = dataset.map(get_seqlen, num_proc=64)

seqlens = np.array([x["seqlen"] for x in ds])

np.save(os.path.join(OUT_DIR, "ldc_seqlens.npy"), seqlens)
