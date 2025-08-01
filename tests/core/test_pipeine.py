import json
import os
import pickle
from collections import defaultdict
from typing import Dict, List, Tuple

os.environ["DCP_DEBUG"] = "DEBUG"

import torch

from dcp.core.cost_model import CommunicationCostModel
from dcp.core.pipeline import (
    check_and_visualize_pipeline,
    generate_pipelines,
)
from dcp.data.dataloader import generate_workload
from dcp.graph_partition.graph import (
    construct_and_partition_graph_multiconstraint,
)
from dcp.utils.logger import logger


def create_problems(
    seqlens,
    n_devices,
    n_nodes,
    block_size,
):
    qkv = torch.randn(sum(seqlens), 3, 12, 128, dtype=torch.bfloat16)
    n_devices_per_node = n_devices // n_nodes
    (
        workload_spec,
        _,
        _,
        _,
        input_q_to_seq_block,
        input_kv_to_seq_block,
        output_qkv_to_seq_block,
        output_lse_to_seq_block,
    ) = generate_workload(
        qkv,
        seqlens,
        block_size,
        n_devices,
        n_devices_per_node,
        "colocate",
    )
    # generate input and output names
    input_to_name = {}
    output_to_name = {}
    for input_id, (
        seq_id,
        block_start,
        block_end,
    ) in input_q_to_seq_block.items():
        input_to_name[input_id] = f"S{seq_id}Q[{block_start}:{block_end}]"
    for input_id, (
        seq_id,
        block_start,
        block_end,
    ) in input_kv_to_seq_block.items():
        input_to_name[input_id] = f"S{seq_id}KV[{block_start}:{block_end}]"
    for output_id, (
        seq_id,
        block_start,
        block_end,
    ) in output_qkv_to_seq_block.items():
        output_to_name[output_id] = f"S{seq_id}QKV[{block_start}:{block_end}]"
    for output_id, (
        seq_id,
        block_start,
        block_end,
    ) in output_lse_to_seq_block.items():
        output_to_name[output_id] = f"S{seq_id}LSE[{block_start}:{block_end}]"
    return (
        workload_spec,
        qkv,
        input_to_name,
        output_to_name,
    )


def generate_triangular_workload_graph(
    block_sizes_per_sequence: List[List[int]],
):
    workloads = []
    input_sizes = []
    output_sizes = []
    work_unit_input_map = []
    work_unit_output_map = []
    colocation_constraints = []
    workload_to_name = {}
    input_to_name = {}
    output_to_name = {}
    for seq_id, block_sizes in enumerate(block_sizes_per_sequence):
        curr_seq_inputs_sizes = [
            block_sizes[i] for i in range(len(block_sizes))
        ] + [block_sizes[i] * 2 for i in range(len(block_sizes))]
        curr_seq_output_sizes = [
            block_sizes[i] for i in range(len(block_sizes))
        ]
        for i in range(len(block_sizes)):
            input_to_name[len(input_sizes) + i] = f"S{seq_id}Q{i}"
        for i in range(len(block_sizes)):
            input_to_name[len(input_sizes) + len(block_sizes) + i] = (
                f"S{seq_id}KV{i}"
            )
        for i in range(len(block_sizes)):
            output_to_name[len(output_sizes) + i] = f"S{seq_id}O{i}"

        for block_idx_q in range(len(block_sizes)):
            for block_idx_kv in range(block_idx_q + 1):
                workloads.append(
                    block_sizes[block_idx_q] * block_sizes[block_idx_kv]
                )
                workload_to_name[len(workloads) - 1] = (
                    f"S{seq_id}Q{block_idx_q}KV{block_idx_kv}"
                )
                work_unit_input_map.append(
                    [
                        len(input_sizes) + block_idx_q,
                        len(input_sizes) + len(block_sizes) + block_idx_kv,
                    ]
                )
                work_unit_output_map.append([len(output_sizes) + block_idx_q])
        colocation_constraints.extend(
            [
                [
                    [
                        len(input_sizes) + i,
                        (len(input_sizes) + len(block_sizes) + i),
                    ],
                    [len(output_sizes) + i],
                ]
                for i in range(len(block_sizes))
            ]
        )
        input_sizes += curr_seq_inputs_sizes
        output_sizes += curr_seq_output_sizes
    return (
        workloads,
        input_sizes,
        output_sizes,
        work_unit_input_map,
        work_unit_output_map,
        colocation_constraints,
        workload_to_name,
        input_to_name,
        output_to_name,
    )


def test_generate_pipelines(
    block_sizes_per_sequence: List[List[int]],
    n_devices: int,
    n_nodes: int,
    intra_node_bandwidth: float,
    inter_node_bandwidth: float,
):
    (
        workloads,
        input_sizes,
        output_sizes,
        work_unit_input_map,
        work_unit_output_map,
        colocation_constraints,
        workload_to_name,
        input_to_name,
        output_to_name,
    ) = generate_triangular_workload_graph(block_sizes_per_sequence)

    (
        device_to_workload_map,
        device_to_input_map,
        device_to_output_map,
    ) = construct_and_partition_graph_multiconstraint(
        n_devices,
        n_nodes,
        workloads,
        input_sizes,
        output_sizes,
        work_unit_input_map,
        work_unit_output_map,
        colocation_constraints=colocation_constraints,
        logger=logger,
    )
    for d, ws in device_to_workload_map.items():
        logger.debug(f"Device {d}: {[workload_to_name[x] for x in ws]}")
    for d, ws in device_to_workload_map.items():
        logger.debug(
            f"Device {d}: total workload {sum([workloads[x] for x in ws])}"
        )
    for d, inputs in device_to_input_map.items():
        logger.debug(
            f"Device {d}: total memory {sum([input_sizes[x] for x in inputs]) + sum([output_sizes[x] for x in device_to_output_map[d]])}"
        )

    input_to_device_map = {}
    for d, inputs in device_to_input_map.items():
        for input in inputs:
            input_to_device_map[input] = d

    # with open("./debug/pipeline_inputs.pkl", "rb") as f:
    #     (workloads,
    #     work_unit_input_map,
    #     input_sizes,
    #     device_to_workload_map,
    #     input_to_device_map,
    #     inter_node_bandwidth,
    #     intra_node_bandwidth,) = pickle.load(f)
    comm_cost_model = CommunicationCostModel(
        0, inter_node_bandwidth, 0, intra_node_bandwidth
    )
    all_devices = [
        (node_id, device_id)
        for node_id in range(n_nodes)
        for device_id in range(n_devices // n_nodes)
    ]
    workloads_per_stage = generate_pipelines(
        all_devices,
        workloads,
        work_unit_input_map,
        input_sizes,
        device_to_workload_map,
        input_to_device_map,
        comm_cost_model,
        logger=logger,
    )
    # check that each workload is assigned to one and only one stage
    workload_to_stage = defaultdict(list)
    for stage_id, stage in enumerate(workloads_per_stage):
        for d, per_stage_workloads in stage.items():
            for w in per_stage_workloads:
                workload_to_stage[w].append(f"Device: {d}, Stage: {stage_id}")
    error = False
    for w, stages in workload_to_stage.items():
        if len(stages) != 1:
            error = True
            logger.error(
                f"Workload {w} is assigned to multiple stages: {stages}"
            )
        if w >= len(workloads):
            error = True
            logger.error(f"Workload {w} does not exist")
    if error:
        # save input for debug
        if not os.path.exists("./debug"):
            os.makedirs("./debug")
        with open("./debug/pipeline_inputs.pkl", "wb") as f:
            pickle.dump(
                (
                    workloads,
                    work_unit_input_map,
                    input_sizes,
                    device_to_workload_map,
                    input_to_device_map,
                    inter_node_bandwidth,
                    intra_node_bandwidth,
                ),
                f,
            )
        raise Exception(
            "Workloads are assigned to multiple stages or do not exist"
        )
    check_and_visualize_pipeline(
        workloads,
        input_sizes,
        work_unit_input_map,
        workloads_per_stage,
        input_to_device_map,
        inter_node_bandwidth,
        intra_node_bandwidth,
    )


def test_generate_pipelines_e2e(
    n_devices,
    n_nodes,
    seqlens,
    block_size,
    intra_node_bandwidth: float,
    inter_node_bandwidth: float,
):
    workload_spec, qkv, input_to_name, output_to_name = create_problems(
        seqlens,
        n_devices,
        n_nodes,
        block_size,
    )
    (
        device_to_workload_map,
        device_to_input_map,
        device_to_output_map,
    ) = construct_and_partition_graph_multiconstraint(
        n_devices,
        n_nodes,
        workload_spec.workloads,
        workload_spec.input_sizes,
        workload_spec.output_sizes,
        workload_spec.work_unit_input_map,
        workload_spec.work_unit_output_map,
        colocation_constraints=workload_spec.colocation_constraints,
        logger=logger,
    )
    for d, ws in device_to_workload_map.items():
        logger.debug(
            f"Device {d}: total workload {sum([workload_spec.workloads[x] for x in ws])}"
        )
    for d, inputs in device_to_input_map.items():
        logger.debug(
            f"Device {d}: total memory {sum([workload_spec.input_sizes[x] for x in inputs]) + sum([workload_spec.output_sizes[x] for x in device_to_output_map[d]])}"
        )
    input_to_device_map = {}
    for d, inputs in device_to_input_map.items():
        for input in inputs:
            input_to_device_map[input] = d

    comm_cost_model = CommunicationCostModel(
        0, inter_node_bandwidth, 0, intra_node_bandwidth
    )
    workloads_per_stage = generate_pipelines(
        workload_spec.workloads,
        workload_spec.work_unit_input_map,
        workload_spec.input_sizes,
        device_to_workload_map,
        input_to_device_map,
        comm_cost_model,
        logger=logger,
    )
    # check that each workload is assigned to one and only one stage
    workload_to_stage = defaultdict(list)
    for stage_id, stage in enumerate(workloads_per_stage):
        for d, per_stage_workloads in stage.items():
            for w in per_stage_workloads:
                workload_to_stage[w].append(f"Device: {d}, Stage: {stage_id}")
    error = False
    for w, stages in workload_to_stage.items():
        if len(stages) != 1:
            error = True
            logger.error(
                f"Workload {w} is assigned to multiple stages: {stages}"
            )
        if w >= len(workload_spec.workloads):
            error = True
            logger.error(f"Workload {w} does not exist")
    if error:
        # save input for debug
        if not os.path.exists("./debug"):
            os.makedirs("./debug")
        with open("./debug/pipeline_inputs.pkl", "wb") as f:
            pickle.dump(
                (
                    workload_spec.workloads,
                    workload_spec.work_unit_input_map,
                    workload_spec.input_sizes,
                    device_to_workload_map,
                    input_to_device_map,
                    inter_node_bandwidth,
                    intra_node_bandwidth,
                ),
                f,
            )
        raise Exception(
            "Workloads are assigned to multiple stages or do not exist"
        )
    check_and_visualize_pipeline(
        workload_spec.workloads,
        workload_spec.input_sizes,
        workload_spec.work_unit_input_map,
        workloads_per_stage,
        input_to_device_map,
        inter_node_bandwidth,
        intra_node_bandwidth,
    )


if __name__ == "__main__":
    # test_generate_pipelines(
    #     block_sizes_per_sequence=[
    #         [4, 4],
    #         [4, 4, 4, 4],
    #         [4, 4, 4, 4, 4, 4, 4],
    #         [4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
    #     ],
    #     n_devices=4,
    #     n_nodes=1,
    #     intra_node_bandwidth=0.25,
    #     inter_node_bandwidth=1.0,
    # )
    test_generate_pipelines_e2e(
        n_devices=4,
        n_nodes=1,
        seqlens=[128, 128, 512, 512, 1024, 2048],
        block_size=64,
        intra_node_bandwidth=500,
        inter_node_bandwidth=100,
    )
