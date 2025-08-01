from __future__ import annotations

__all__ = ["greedy_selection_per_device"]

def greedy_selection_per_device(
    devices: set[tuple[int, int]],
    scheduled_workloads: set[int],
    workload_costs: list[float],
    input_sizes: list[float],
    workload_input_map: list[list[int]],
    input_block_type_map: list[str],
    device_workload_map: dict[tuple[int, int], list[int]],
    device_local_inputs_map: dict[tuple[int, int], dict[int, int]],
    target_comm_load_per_node: dict[tuple[int, int], float],
    stage_id: int,
) -> tuple[
    dict[tuple[int, int], set[int]],
    dict[tuple[int, int], list[tuple[int, tuple[int, int]]]],
    dict[tuple[int, int], dict[int, int]],
]: ...
