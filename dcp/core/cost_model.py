from __future__ import annotations

from typing import Dict, List, Tuple

import torch

from dcp.core.instructions import AttnBackwardInstr, AttnInstr
from dcp.core.serialization import deserialize_float, serialize_float


class CostModel:
    def get_cost(self, *args, **kwargs):
        # cost should be returned in ms
        raise NotImplementedError()


class CommunicationCostModel(CostModel):
    def __init__(
        self,
        inter_node_latency: float = 0,
        inter_node_bandwidth: float = 400,
        intra_node_latency: float = 0,
        intra_node_bandwidth: float = 4800,
    ):
        # latency is in ms, bandwidth is in Gbps
        self.inter_node_latency = inter_node_latency
        self.inter_node_bandwidth = inter_node_bandwidth
        self.intra_node_latency = intra_node_latency
        self.intra_node_bandwidth = intra_node_bandwidth

    def __repr__(self):
        return "CommunicationCostModel({},{},{},{})".format(
            self.inter_node_latency,
            self.inter_node_bandwidth,
            self.intra_node_latency,
            self.intra_node_bandwidth,
        )

    def get_cost(
        self, src: Tuple[int, int], dst: Tuple[int, int], size: int
    ) -> float:
        # size is in bytes
        if src[0] == dst[0]:
            if src[1] == dst[1]:
                return 0.0
            return (
                self.intra_node_latency
                + (size * 8 / 1e6) / self.intra_node_bandwidth
            )
        else:
            return (
                self.inter_node_latency
                + (size * 8 / 1e6) / self.inter_node_bandwidth
            )

    def serialize(self) -> bytes:
        return b"".join(
            [
                serialize_float(x)
                for x in [
                    self.inter_node_latency,
                    self.inter_node_bandwidth,
                    self.intra_node_latency,
                    self.intra_node_bandwidth,
                ]
            ]
        )

    @classmethod
    def deserialize(cls, bytes: bytes):
        inter_node_latency, bytes = deserialize_float(bytes)
        inter_node_bandwidth, bytes = deserialize_float(bytes)
        intra_node_latency, bytes = deserialize_float(bytes)
        intra_node_bandwidth, bytes = deserialize_float(bytes)

        return (
            cls(
                inter_node_latency,
                inter_node_bandwidth,
                intra_node_latency,
                intra_node_bandwidth,
            ),
            bytes,
        )


class AttnRooflineCostModel(CostModel):
    def __init__(self, max_tflops: float = 280, max_mem_bw: float = 1500):
        """
        Args:
            max_tflops: peak TFLOPS/s for a GPU
            max_mem_bw: Maximum memory bandwidth in GB/s
        """
        self.max_tflops = max_tflops
        self.max_mem_bw = max_mem_bw

    def __repr__(self):
        return "AttnRooflineCostModel({},{})".format(
            self.max_tflops, self.max_mem_bw
        )

    def __eq__(self, other: "AttnRooflineCostModel") -> bool:
        return (
            self.max_tflops == other.max_tflops
            and self.max_mem_bw == other.max_mem_bw
        )

    def _get_flops(
        self,
        seqlens_q: List[int],
        seqlens_k: List[int],
        nheads: int,
        headdim: int,
        mask_sparsity: float,
        mode: str = "fwd",
    ) -> float:
        # modified from Dao-AILab/flash-attention
        # benchmarks/benchmark_flash_attention.py
        assert mode in ["fwd", "bwd"]
        f = (
            4
            * sum(q * k for q, k in zip(seqlens_q, seqlens_k))
            * nheads
            * headdim
            * mask_sparsity
        )
        return f if mode == "fwd" else 2.5 * f

    def _get_mem_access(
        self,
        seqlens_q: List[int],
        seqlens_k: List[int],
        n_query_groups: int,
        headdim: int,
        mask_sparsity: float,
        mode: str = "fwd",
    ) -> float:
        total_mem_access = 0
        token_size = 2 * n_query_groups * headdim
        if mode == "fwd":
            m_block_size = 64
            for q_seqlen, k_seqlen in zip(seqlens_q, seqlens_k):
                # all kv loaded once for each m block
                total_mem_access += (
                    (q_seqlen + m_block_size - 1)
                    // m_block_size
                    * k_seqlen
                    * token_size
                )
        else:
            n_block_size = 128  # a rough estimate
            # load dO once to compute rowsum(dO * O)
            dO_size = sum(seqlens_q) * token_size
            total_mem_access += dO_size
            for q_seqlen, k_seqlen in zip(seqlens_q, seqlens_k):
                # load q, o, do and dq once for each n block
                total_mem_access += (
                    4
                    * q_seqlen
                    * (k_seqlen + n_block_size - 1)
                    // n_block_size
                    * token_size
                )
        return total_mem_access * mask_sparsity

    def get_cost(
        self,
        seqlens_q,
        seqlens_k,
        mask,
        n_heads,
        n_query_groups,
        headdim,
        is_forward=True,
    ):
        # calculate attn mask sparsity
        actual_elems = 0
        total_elems = 0
        for seqlen_q, seqlen_k in zip(seqlens_q, seqlens_k):
            total_elems += seqlen_q * seqlen_k
        if mask.dim() == 2:
            # one range masking
            actual_elems += torch.clamp(mask[1] - mask[0], min=0).sum().item()
        elif mask.dim() == 3:
            # two range masking
            # now mask is a 3-d tensor
            range_1_elems = (
                torch.clamp(mask[0][1] - mask[0][0], min=0).sum().item()
            )
            range_2_elems = (
                torch.clamp(mask[1][1] - mask[1][0], min=0).sum().item()
            )
            overlap_elems = (
                torch.clamp(
                    torch.minimum(mask[0][1], mask[1][1])
                    - torch.maximum(mask[0][0], mask[1][0]),
                    min=0,
                )
                .sum()
                .item()
            )
            actual_elems = range_1_elems + range_2_elems - overlap_elems

        else:
            raise ValueError(f"Invalid mask shape: {mask.shape}")

        mask_sparsity = actual_elems / total_elems
        if is_forward:
            flops = self._get_flops(
                seqlens_q, seqlens_k, n_heads, headdim, mask_sparsity
            )
            mem_access = self._get_mem_access(
                seqlens_q, seqlens_k, n_query_groups, headdim, mask_sparsity
            )
        else:
            flops = self._get_flops(
                seqlens_q,
                seqlens_k,
                n_heads,
                headdim,
                mask_sparsity,
                mode="bwd",
            )
            mem_access = self._get_mem_access(
                seqlens_q,
                seqlens_k,
                n_heads,
                headdim,
                mask_sparsity,
                mode="bwd",
            )
        flops_time = flops / self.max_tflops * 1e-9
        mem_time = mem_access / self.max_mem_bw * 1e-6
        return flops_time, mem_time

    def serialize(self) -> bytes:
        return b"".join(
            [serialize_float(x) for x in [self.max_tflops, self.max_mem_bw]]
        )

    @classmethod
    def deserialize(cls, bytes: bytes):
        max_tflops, bytes = deserialize_float(bytes)
        max_mem_bw, bytes = deserialize_float(bytes)
        return cls(max_tflops, max_mem_bw), bytes
