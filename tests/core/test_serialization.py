import pytest
import torch
import torch.nn.functional as F

from dcp.core.block_table import WorkloadSpec
from dcp.core.cost_model import AttnRooflineCostModel
from dcp.data.dataloader import _generate_workload


def test_serialize_workload_spec():
    raw_seqlens = [123, 111, 111]
    attn_mask = torch.zeros(sum(raw_seqlens), 2, dtype=torch.int32)
    cu_seqlens = F.pad(
        torch.cumsum(torch.tensor(raw_seqlens, dtype=torch.int32), 0), (1, 0)
    ).to(torch.int32)
    for seq_id, seqlen in enumerate(raw_seqlens):
        attn_mask[cu_seqlens[seq_id] : cu_seqlens[seq_id + 1], 0] = 0
        attn_mask[cu_seqlens[seq_id] : cu_seqlens[seq_id + 1], 1] = (
            torch.arange(1, seqlen + 1)
        )
    workload_spec = _generate_workload(
        (sum(raw_seqlens), 3, 8, 128),
        torch.bfloat16,
        raw_seqlens,
        256,
        1,
        4,
        8,
        AttnRooflineCostModel(),
        attn_mask,
    )
    serialized = workload_spec.serialize()
    deserialized, _ = WorkloadSpec.deserialize(serialized)
    deserialized: WorkloadSpec
    assert workload_spec.workloads == deserialized.workloads
    assert (
        workload_spec.work_unit_input_map == deserialized.work_unit_input_map
    )
    assert (
        workload_spec.work_unit_output_map == deserialized.work_unit_output_map
    )
    assert workload_spec.block_mapping == deserialized.block_mapping

    assert (
        workload_spec.input_to_device_map == deserialized.input_to_device_map
    )
    assert (
        workload_spec.output_to_device_map == deserialized.output_to_device_map
    )
    assert workload_spec.work_to_device_map == deserialized.work_to_device_map
    assert workload_spec.work_to_stage_map == deserialized.work_to_stage_map
    assert (
        workload_spec.colocation_constraints
        == deserialized.colocation_constraints
    )


if __name__ == "__main__":
    test_serialize_workload_spec()
