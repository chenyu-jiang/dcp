from math import sqrt
import numpy as np

SEQLENS_PATH = "./longalign_seqlens.npy"

with open(SEQLENS_PATH, "rb") as f:
    seqlens = np.load(f)


def scale_mean(seqlens, multiplier=1.0):
    new_seqlens = [int(round(multiplier * seqlen)) for seqlen in seqlens]
    return np.array(new_seqlens)


# variance_scales = [0.25, 0.5, 1, 2, 4]
mean_scales = [0.5, 1, 2, 4]
for scale in mean_scales:
    new_seqlens = scale_mean(seqlens, scale)
    new_seqlens_path = f"./longalign_seqlens_mean_{scale}.npy"
    with open(new_seqlens_path, "wb") as f:
        np.save(f, new_seqlens)
    print(f"Saved new seqlens to {new_seqlens_path}")
