import os

# os.environ['HF_HOME'] = '/opt/dlami/nvme/huggingface'

from datasets import load_dataset
import numpy as np

import tiktoken

OUT_DIR = "./"

dataset = load_dataset("THUDM/LongAlign-10k", split="train", num_proc=64)
encoding = tiktoken.get_encoding("r50k_base")


def get_seqlen(sample):
    text = ""
    for conversation in sample["messages"]:
        # we just randomly concat then since this will not
        # greatly affect seqlen
        text += 'role: "' + conversation["role"] + '", '
        text += 'content: "' + conversation["content"] + '".'
    tokenized = encoding.encode(text, allowed_special="all")
    sample["seqlen"] = len(tokenized)
    return sample


ds = dataset.map(get_seqlen, num_proc=64)

seqlens = np.array([x["seqlen"] for x in ds])

np.save(os.path.join(OUT_DIR, "longalign_seqlens.npy"), seqlens)
