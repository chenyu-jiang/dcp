from dataclasses import dataclass
from typing import Union

from dcp.core.cost_model import (
    AttnRooflineCostModel,
    CommunicationCostModel,
)
from dcp.core.serialization import (
    deserialize_list_of_ints,
    serialize_list_of_ints,
)


@dataclass
class ExecutionContext:
    n_devices_per_node: int
    n_nodes: int
    comm_cost_model: CommunicationCostModel
    comp_cost_model: AttnRooflineCostModel

    def serialize(self) -> bytes:
        return b"".join(
            [
                serialize_list_of_ints(
                    [
                        self.n_devices_per_node,
                        self.n_nodes,
                    ]
                ),
                self.comm_cost_model.serialize(),
                self.comp_cost_model.serialize(),
            ]
        )

    @classmethod
    def deserialize(cls, data: bytes):
        (
            n_devices_per_node,
            n_nodes,
        ), data = deserialize_list_of_ints(data)
        comm_cost_model, data = CommunicationCostModel.deserialize(data)
        comp_cost_model, data = AttnRooflineCostModel.deserialize(data)

        return (
            cls(n_devices_per_node, n_nodes, comm_cost_model, comp_cost_model),
            data,
        )


def get_default_execution_context(
    n_devices: int, n_nodes: int
) -> ExecutionContext:
    return ExecutionContext(
        n_devices // n_nodes,
        n_nodes,
        CommunicationCostModel(),
        AttnRooflineCostModel(),
    )


@dataclass
class ModelSpec:
    # Default setting:
    #  * mlp_hidden_size = 4x hidden_dim
    #  * kv_channels = hidden_dim // num_attn_heads
    #  * use FP16 mixed precision training with Adam optimizer.
    head_dim: int
    n_heads: int
    n_query_groups: int = 1

    def serialize(self) -> bytes:
        return serialize_list_of_ints(
            [
                self.head_dim,
                self.n_heads,
                self.n_query_groups,
            ]
        )

    @classmethod
    def deserialize(cls, data: bytes):
        (
            head_dim,
            n_heads,
            n_query_groups,
        ), data = deserialize_list_of_ints(data)

        return (
            cls(
                head_dim,
                n_heads,
                n_query_groups,
            ),
            data,
        )
