import argparse
from collections import defaultdict

import numpy as np
import torch

from dcp.core.block_table import WorkloadSpec
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


def get_reference_comm_volume(
    qkv: torch.Tensor,
    n_devices: int,
    n_nodes: int = 1,
):
    total_per_device_kv_size = qkv.numel() / qkv.shape[-1] / n_devices / 3 * 2
    total_per_device_volume = (n_devices - 1) * total_per_device_kv_size
    total_volume = total_per_device_volume * n_devices
    inter_node_volume = (n_nodes - 1) * total_per_device_volume
    intra_node_volume = total_volume - inter_node_volume
    return inter_node_volume, intra_node_volume


def generate_sequence_lengths(n_seqs, min_seqlen, max_seqlen):
    seqlens = np.random.uniform(min_seqlen, max_seqlen, size=n_seqs).astype(
        int
    )
    return seqlens.tolist()


def get_comm_volume(
    workload_spec: WorkloadSpec,
    device_to_workload_map,
    device_to_input_map,
    device_to_output_map,
):
    inter_node_cost = 0
    intra_node_cost = 0
    unique_device_per_input = defaultdict(set)
    unique_device_per_output = defaultdict(set)
    for device, work_unit_ids in device_to_workload_map.items():
        for work_unit_id in work_unit_ids:
            for input_id in workload_spec.work_unit_input_map[work_unit_id]:
                unique_device_per_input[input_id].add(device)
            for output_id in workload_spec.work_unit_output_map[work_unit_id]:
                unique_device_per_output[output_id].add(device)
    input_to_device_map = {}
    output_to_device_map = {}
    for device, input_ids in device_to_input_map.items():
        for input_id in input_ids:
            input_to_device_map[input_id] = device
    for device, output_ids in device_to_output_map.items():
        for output_id in output_ids:
            output_to_device_map[output_id] = device
    # calculate communication volume
    for input_id, devices in unique_device_per_input.items():
        assigned_device = input_to_device_map[input_id]
        assigned_nodes = set()
        for device in devices:
            assigned_nodes.add(device[0])
        assigned_device_per_node = defaultdict(set)
        for device in devices:
            assigned_device_per_node[device[0]].add(device[1])
        if assigned_device[0] in assigned_nodes:
            inter_node_cost += (
                len(assigned_nodes) - 1
            ) * workload_spec.input_sizes[input_id]
        else:
            inter_node_cost += (
                len(assigned_nodes) * workload_spec.input_sizes[input_id]
            )
        for node, devices in assigned_device_per_node.items():
            if (
                node == assigned_device[0]
                and assigned_device[1] not in devices
            ):
                intra_node_cost += workload_spec.input_sizes[input_id]
            else:
                intra_node_cost += (
                    len(devices) - 1
                ) * workload_spec.input_sizes[input_id]
    for output_id, devices in unique_device_per_output.items():
        assigned_nodes = set()
        for device in devices:
            assigned_nodes.add(device[0])
        assigned_device_per_node = defaultdict(set)
        for device in devices:
            assigned_device_per_node[device[0]].add(device[1])
        inter_node_cost += (
            len(assigned_nodes) - 1
        ) * workload_spec.output_sizes[output_id]
        for node, devices in assigned_device_per_node.items():
            intra_node_cost += (len(devices) - 1) * workload_spec.output_sizes[
                output_id
            ]
    return inter_node_cost, intra_node_cost


def main(
    n_devices,
    n_nodes,
    seqlens,
    formulation,
    block_size,
    epsilon=0.01,
    mode="default",
    colocation_constraint_hardness="soft",
):
    workload_spec, qkv, input_to_name, output_to_name = create_problems(
        seqlens,
        n_devices,
        n_nodes,
        block_size,
    )
    if formulation == "hypergraph":
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
            input_to_device_map=workload_spec.input_to_device_map,
            output_to_device_map=workload_spec.output_to_device_map,
            colocation_constraint_type=colocation_constraint_hardness,
            epsilon=epsilon,
            mode=mode,
            logger=logger,
        )
        input_to_device_map = {}
        output_to_device_map = {}
        for device, input_ids in device_to_input_map.items():
            for input_id in input_ids:
                input_to_device_map[input_id] = device
        for device, output_ids in device_to_output_map.items():
            for output_id in output_ids:
                output_to_device_map[output_id] = device
        inter_node_cost, intra_node_cost = get_comm_volume(
            workload_spec,
            device_to_workload_map,
            device_to_input_map,
            device_to_output_map,
        )
    else:
        raise ValueError(f"Formulation {formulation} not supported")
    logger.info(f"Inter node cost: {inter_node_cost}")
    logger.info(f"Intra node cost: {intra_node_cost}")
    # check all workloads are assigned
    workload_assigned = set()
    for device, work_unit_ids in device_to_workload_map.items():
        workload_assigned.update(work_unit_ids)
    assert workload_assigned == set(range(len(workload_spec.workloads)))
    per_device_workload = {
        device: sum(workload_spec.workloads[w] for w in work_unit_ids)
        for device, work_unit_ids in device_to_workload_map.items()
    }
    for device, workload in per_device_workload.items():
        logger.info(f"Device {device} workload: {workload}")
    logger.info(
        "Device workload imbalance: {}".format(
            max(per_device_workload.values())
            / (sum(per_device_workload.values()) / len(per_device_workload))
        )
    )
    # check all input and output are assigned
    # assert set(range(len(workload_spec.input_sizes))) == set(input_to_device_map.keys())
    for input_id in range(len(workload_spec.input_sizes)):
        if input_id not in input_to_device_map:
            raise ValueError(f"Input {input_id} is not assigned to any device")
    assert set(range(len(workload_spec.output_sizes))) == set(
        output_to_device_map.keys()
    )
    per_device_memory = {device: 0 for device in device_to_workload_map.keys()}
    for input_id, device in input_to_device_map.items():
        per_device_memory[device] += workload_spec.input_sizes[input_id]
    for output_id, device in output_to_device_map.items():
        per_device_memory[device] += workload_spec.output_sizes[output_id]
    for device, memory in per_device_memory.items():
        logger.info(f"Device {device} memory: {memory}")
    logger.info(
        "Device memory imbalance: {}".format(
            max(per_device_memory.values())
            / (sum(per_device_memory.values()) / len(per_device_workload))
        )
    )
    inter_node_volume, intra_node_volume = get_reference_comm_volume(
        qkv, n_devices, n_nodes
    )
    if n_nodes > 1:
        logger.info(
            f"Inter node volume ratio: {inter_node_cost / inter_node_volume}"
        )
    logger.info(
        f"Intra node volume ratio: {intra_node_cost / intra_node_volume}",
    )
    # check if colocation constraints are satisfied
    violated_colocation_constraints = []
    for input_ids, output_ids in workload_spec.colocation_constraints:
        device = None
        violated = False
        for input_id in input_ids:
            if device is None:
                device = input_to_device_map[input_id]
            elif device != input_to_device_map[input_id]:
                violated = True
                break
        if not violated:
            for output_id in output_ids:
                if device != output_to_device_map[output_id]:
                    violated = True
                    break
        if violated:
            violated_colocation_constraints.append((input_ids, output_ids))
    logger.info(
        f"Number of violated colocation constraints: {len(violated_colocation_constraints)} / {len(workload_spec.colocation_constraints)}"
    )
    for idx, (input_ids, output_ids) in enumerate(
        violated_colocation_constraints
    ):
        logger.info(f"Constraint {idx}:")
        for input_id in input_ids:
            logger.info(
                f"\t{input_to_name[input_id]} -> Device {input_to_device_map[input_id]}"
            )
        for output_id in output_ids:
            logger.info(
                f"\t{output_to_name[output_id]} -> Device {output_to_device_map[output_id]}"
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d", "--n-devices", type=int, default=16, help="number of devices"
    )
    parser.add_argument(
        "-n", "--n-nodes", type=int, default=2, help="number of nodes"
    )
    parser.add_argument(
        "-ns", "--n-seqs", type=int, default=8, help="number of sequences"
    )
    parser.add_argument(
        "-min",
        "--min-seqlen",
        type=int,
        default=128,
        help="minimum sequence length",
    )
    parser.add_argument(
        "-max",
        "--max-seqlen",
        type=int,
        default=1024,
        help="maximum sequence length",
    )
    parser.add_argument(
        "-e",
        "--epsilon",
        type=float,
        default=0.02,
        help="epsilon value for partitioning",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument(
        "-f",
        "--formulation",
        type=str,
        choices=["hypergraph", "graph"],
        default="graph",
        help="formulation type.",
    )
    parser.add_argument(
        "-b",
        "--block-size",
        type=int,
        default=None,
        help="use fixed block size for all sequences",
    )
    parser.add_argument(
        "-m",
        "--partition-mode",
        type=str,
        choices=["default", "speed", "quality"],
        default="default",
        help="PaToH partition mode",
    )
    parser.add_argument(
        "-cc",
        "--colocation-constraint-hardness",
        type=str,
        choices=["none", "soft", "hard"],
        default="soft",
        help="colocation constraint hardness",
    )

    args = parser.parse_args()

    # seqlens = generate_sequence_lengths(
    #     args.n_seqs, args.min_seqlen, args.max_seqlen
    # )
    seqlens = [128, 128, 512, 512, 1024, 2048]
    logger.info(f"Sequence lengths: {seqlens}")
    main(
        args.n_devices,
        args.n_nodes,
        seqlens,
        args.formulation,
        args.block_size,
        args.epsilon,
        args.partition_mode,
        args.colocation_constraint_hardness,
    )
