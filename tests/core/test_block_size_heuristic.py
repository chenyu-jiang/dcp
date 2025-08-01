# from dcp.data.dataloader import _block_size_heuristic
from typing import List
import math


def _block_size_heuristic(
    n_devices: int, raw_seqlens: List[int], mem_epsilon: float
):
    def _get_ref_imbalance(block_size):
        blocks = []
        for seqlen in raw_seqlens:
            if seqlen >= block_size:
                blocks.extend([block_size] * (seqlen // block_size))
            if seqlen % block_size > 0:
                blocks.append(seqlen % block_size)
        # try to evenly distribute blocks across devices
        # use greedy algorithm to get a reference
        per_device_sizes = [0] * n_devices
        for block in sorted(blocks, reverse=True):
            min_idx = per_device_sizes.index(min(per_device_sizes))
            per_device_sizes[min_idx] += block
        # ref_imbalance = max(per_device_sizes) / (
        #     sum(per_device_sizes) / len(per_device_sizes)
        # )
        ref_imbalance = (
            max(per_device_sizes) / min(per_device_sizes)
            if min(per_device_sizes) > 0
            else float("inf")
        )
        print(
            f"Block size: {block_size}, Per device sizes: {per_device_sizes}, Ref imbalance: {ref_imbalance}, number of blocks: {len(blocks)}"
        )
        return ref_imbalance

    block_size = max(raw_seqlens)
    block_size = 2 ** math.ceil(math.log2(block_size))

    while (
        _get_ref_imbalance(block_size) > (1 + mem_epsilon) and block_size > 256
    ):
        block_size //= 2
    return max(256, block_size)


block_size = _block_size_heuristic(8, [11610, 12200, 16384, 16384], 0.2)

print(block_size)
