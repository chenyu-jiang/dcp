import json

import numpy as np

MAX_SEQLEN = 131072
N_TOKENS_PER_BATCH = 131072

for mean in [0.5, 1, 2, 4]:
    seqlens = np.load(f"./longalign_seqlens_mean_{mean}.npy")
    random_seed = 42
    np.random.seed(random_seed)
    # shuffle the seqlens
    np.random.shuffle(seqlens)

    seqlens = [int(seqlen) for seqlen in seqlens if seqlen <= MAX_SEQLEN]

    batches = []

    curr_batch = []
    curr_batch_sum = 0
    for seqlen in seqlens:
        if curr_batch_sum + seqlen > N_TOKENS_PER_BATCH:
            if curr_batch_sum >= N_TOKENS_PER_BATCH * 0.75:
                batches.append(curr_batch)
            curr_batch = []
            curr_batch_sum = 0
        curr_batch.append(seqlen)
        curr_batch_sum += seqlen
    # drop the last batch
    config = {
        "batches": batches,
        "max_dp_degree": N_TOKENS_PER_BATCH // int(np.max(seqlens)),
    }
    print("max dp degree: ", N_TOKENS_PER_BATCH // int(np.max(seqlens)))
    print("max seqlen: ", np.max(seqlens))
    print("mean seqs per batch", np.mean([len(batch) for batch in batches]))
    print("min seqs per batch", np.min([len(batch) for batch in batches]))
    print("max seqs per batch", np.max([len(batch) for batch in batches]))
    print("num batches", len(batches))
    with open(f"./longalign_{N_TOKENS_PER_BATCH}_mean{mean}.json", "w") as f:
        f.write(json.dumps(config))
