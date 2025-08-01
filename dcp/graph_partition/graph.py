import logging
import os
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import kahypar
import mtkahypar
import networkx as nx

# import pymetis
import pypatoh

from dcp.core.block_table import WorkloadSpec, BlockType
from dcp.core.common import ExecutionContext
from dcp.utils.logger import Logger, read_env_bool

GP_LOGGING_ENABLED = read_env_bool("DCP_LOG_GRAPH_PARTITION", default=False)
N_CONSTRS = 2


class HGPSolver:
    PATOH: str = "patoh"
    KAHYPAR: str = "kahypar"
    MTKAHYPAR: str = "mtkahypar"


def construct_and_partition_graph_multiconstraint(
    context: ExecutionContext,
    workload: WorkloadSpec,
    mem_epsilon: float = 0.2,
    comp_epsilon: float = 0.2,
    inter_node_comp_epsilon_factor: float = 5.0,
    mode: str = "default",
    rounding_scale: int = 1,
    workload_scale: int = -1,
    solver: str = HGPSolver.PATOH,
    improve_upon_heuristic: bool = False,
    num_vcycles: int = 20,
    enforce_comm_balance: bool = False,
    comm_epsilon: float = 0.05,
    logger: Optional[Logger] = None,
):
    # Construct a hypergraph describing workloads and input/output dependencies
    # Nodes: work units + (optional) one dummy node for each device.
    #        (with weight 1). The dummy nodes are fixed to the partition
    #        corresponding to the device they are assigned to.
    # Hyperedges/Nets:
    #   - input dependencies: one hyperedge per input unit, connecting work
    #    units that consume the same input
    #   - output dependencies: one hyperedge per output unit, connecting work
    #    units that contribute to the same output
    #   - assignment: if specified, one hyperedge per input/output unit,
    #    connecting the unit to the dummy node corresponding to the device
    #    it is assigned to.

    logging_enabled = logger is not None and GP_LOGGING_ENABLED

    assert (
        not workload.input_to_device_map
    ), "fixing input to device is not supported"
    assert (
        not workload.output_to_device_map
    ), "fixing output to device is not supported"

    n_total_devices = context.n_devices_per_node * context.n_nodes

    if logging_enabled:
        logger.debug(
            f"Received input with total {len(workload.workloads)} work units, "
            f"{len(workload.input_sizes)} input units, "
            f"{len(workload.output_sizes)} output units, "
            f"{n_total_devices} devices, "
            f"{context.n_nodes} nodes."
        )
    # calculate max input weight for rounding
    init_vertex_sizes = []
    for input_ids, output_ids in workload.colocation_constraints:
        curr_vertex_size = 0
        for input_id in input_ids:
            curr_vertex_size += workload.input_sizes[input_id]
        for output_id in output_ids:
            curr_vertex_size += workload.output_sizes[output_id]
        init_vertex_sizes.append(curr_vertex_size)
    max_weight = max(init_vertex_sizes)
    # max_weight = max(max(workload.input_sizes), max(workload.output_sizes))
    # if max_weight > 1e7:
    #     weight_rounding_scale = 1e-6
    # elif max_weight > 1e4:
    #     weight_rounding_scale = 1e-3
    # else:
    #     weight_rounding_scale = 1
    weight_rounding_scale = 45000 / max_weight
    if logging_enabled:
        logger.debug(f"Weight rounding scale: {weight_rounding_scale}")

    def _round_net_weight(weight):
        return max(int(weight * weight_rounding_scale * rounding_scale), 1)

    def _construct_graph_for_work_ids(
        k,
        from_work_units,
        current_node_id=None,
        input_to_node_map=None,
        output_to_node_map=None,
    ):
        if logging_enabled:
            logger.debug("==" * 10 + " Constructing hypergraph " + "==" * 10)
        # if current_node_id is None, then we are constructing the node
        # level graph, otherwise we are constructing the intra-node graph
        pins = []
        xpins = [0]
        net_weights = []
        vertex_weights = []

        # aux info for epsilon scaling
        mem_weights = []

        # map from original id to node index used in the hypergraph
        # data block ids that share the same colocation constraints are modeled
        # as a single node in the hypergraph
        work_id_to_vertex_id = {}
        input_id_to_vertex_id = {}
        output_id_to_vertex_id = {}
        input_id_to_net_id = {}
        output_id_to_net_id = {}

        input_data_unit_to_work_units = defaultdict(list)
        output_data_unit_to_work_units = defaultdict(list)
        for work_unit in from_work_units:
            input_units = workload.work_unit_input_map[work_unit]
            for input_unit in input_units:
                input_data_unit_to_work_units[input_unit].append(work_unit)
        for work_unit in from_work_units:
            output_units = workload.work_unit_output_map[work_unit]
            for output_unit in output_units:
                output_data_unit_to_work_units[output_unit].append(work_unit)

        # reassign work ids
        nonlocal workload_scale
        if workload_scale == -1:
            # min_workload = min([w for w in workload.workloads if w > 1e-4])
            max_workload = max(
                [workload.workloads[w] for w in from_work_units]
            )
            # sum_workload = sum([workload.workloads[w] for w in from_work_units])
            # assert min_workload > 0, "workload must be positive"
            # diff = max_workload - min_workload
            # if diff < 1e6:
            #     workload_scale = 100 / min_workload
            # else:
            #     workload_scale = 100 / diff
            workload_scale = 10000 / max_workload
            if logging_enabled:
                logger.debug(
                    f"Set workload scale factor: {workload_scale}, "
                    # f"min workload: {min_workload}, after scaling: {min_workload * workload_scale}, "
                    f"max workload: {max_workload}, after scaling: {max_workload * workload_scale}"
                )
        # compute memory weight scale
        # so that min memory weight > sum(scaled_compute_weight)
        sum_scaled_comp_weight = sum(
            [
                max(1, int(workload.workloads[work_unit] * workload_scale))
                for work_unit in from_work_units
            ]
        )
        min_memory = min(init_vertex_sizes)
        memory_scale = (sum_scaled_comp_weight / min_memory) * 100
        if memory_scale < 1:
            memory_scale = 1
        if logging_enabled:
            logger.debug(
                f"Sum scaled compute weight: {sum_scaled_comp_weight}"
            )
            logger.debug(
                f"Current graph memory scale factor: {memory_scale}, "
                f"min memory: {min_memory}, after scaling: {min_memory * memory_scale}"
            )

        def _round_memory_weight(weight):
            return max(int(weight * memory_scale), 1)

        for idx, work_unit in enumerate(from_work_units):
            work_id_to_vertex_id[work_unit] = idx
            if solver in [HGPSolver.KAHYPAR, HGPSolver.MTKAHYPAR]:
                # KAHYPAR do not support 2d weights, we merge memory
                # and compute weight by scaling, i.e. memory weight
                # = memory weight + max compute weight.
                # We can't enforce epsilon on both memory and compute
                # in this way, so we scale epsilon to work only on compute
                # weight (while memory should always be perfectly balanced).
                vertex_weights.extend(
                    [
                        max(
                            1,
                            int(
                                workload.workloads[work_unit] * workload_scale
                            ),
                        )
                    ]
                )
            elif solver == HGPSolver.PATOH:
                # PATOH supports 2d weights
                vertex_weights.extend(
                    [
                        max(
                            1,
                            int(
                                workload.workloads[work_unit] * workload_scale
                            ),
                        ),
                        0,
                    ]
                )
        compute_weights = vertex_weights.copy()
        # reassign input/output ids
        for input_unit in input_data_unit_to_work_units.keys():
            if input_unit in input_id_to_vertex_id:
                continue
            if solver in [HGPSolver.KAHYPAR, HGPSolver.MTKAHYPAR]:
                new_vertex_id = len(vertex_weights)
            else:
                new_vertex_id = len(vertex_weights) // 2
            input_id_to_vertex_id[input_unit] = new_vertex_id
            # get all input and output ids that share the same location
            total_mem_size = workload.input_sizes[input_unit]
            for co_input_ids, co_output_ids in workload.colocation_constraints:
                if input_unit in co_input_ids:
                    for co_input_id in co_input_ids:
                        if co_input_id == input_unit:
                            continue
                        total_mem_size += workload.input_sizes[co_input_id]
                        assert co_input_id not in input_id_to_vertex_id
                        input_id_to_vertex_id[co_input_id] = new_vertex_id
                    for co_output_id in co_output_ids:
                        total_mem_size += workload.output_sizes[co_output_id]
                        assert co_output_id not in output_id_to_vertex_id
                        output_id_to_vertex_id[co_output_id] = new_vertex_id
            if (
                current_node_id is not None
                and input_unit in input_to_node_map
                and input_to_node_map[input_unit] != current_node_id
            ):
                # this input is transferred from/to another node
                # we don't need to consider its location constraints
                if solver in [HGPSolver.KAHYPAR, HGPSolver.MTKAHYPAR]:
                    # Kahypar does not like 0 weight, use 1 instead
                    vertex_weights.extend([1])
                elif solver == HGPSolver.PATOH:
                    vertex_weights.extend([0, 0])
            else:
                if solver in [HGPSolver.KAHYPAR, HGPSolver.MTKAHYPAR]:
                    vertex_weights.extend(
                        [_round_memory_weight(total_mem_size)]
                    )
                    mem_weights.append(_round_memory_weight(total_mem_size))
                elif solver == HGPSolver.PATOH:
                    vertex_weights.extend(
                        [0, _round_memory_weight(total_mem_size)]
                    )
        for output_unit in output_data_unit_to_work_units.keys():
            if output_unit in output_id_to_vertex_id:
                continue
            if solver in [HGPSolver.KAHYPAR, HGPSolver.MTKAHYPAR]:
                new_vertex_id = len(vertex_weights)
            elif solver == HGPSolver.PATOH:
                new_vertex_id = len(vertex_weights) // 2
            output_id_to_vertex_id[output_unit] = new_vertex_id
            # get all input and output ids that share the same location
            total_mem_size = workload.output_sizes[output_unit]
            for co_input_ids, co_output_ids in workload.colocation_constraints:
                if output_unit in co_output_ids:
                    for co_input_id in co_input_ids:
                        total_mem_size += workload.input_sizes[co_input_id]
                        assert co_input_id not in input_id_to_vertex_id
                        input_id_to_vertex_id[co_input_id] = new_vertex_id
                    for co_output_id in co_output_ids:
                        if co_output_id == output_unit:
                            continue
                        total_mem_size += workload.output_sizes[co_output_id]
                        assert co_output_id not in output_id_to_vertex_id
                        output_id_to_vertex_id[co_output_id] = new_vertex_id
            if (
                current_node_id is not None
                and output_unit in output_to_node_map
                and output_to_node_map[output_unit] != current_node_id
            ):
                # this output is transferred from/to another node
                # we don't need to consider its location constraints
                if solver in [HGPSolver.KAHYPAR, HGPSolver.MTKAHYPAR]:
                    vertex_weights.extend([1])
                elif solver == HGPSolver.PATOH:
                    vertex_weights.extend([0, 0])
            else:
                if solver in [HGPSolver.KAHYPAR, HGPSolver.MTKAHYPAR]:
                    vertex_weights.extend(
                        [_round_memory_weight(total_mem_size)]
                    )
                    mem_weights.append(_round_memory_weight(total_mem_size))
                elif solver == HGPSolver.PATOH:
                    vertex_weights.extend(
                        [0, _round_memory_weight(total_mem_size)]
                    )
        # create hyperedges
        for input_unit, work_units in input_data_unit_to_work_units.items():
            input_id_to_net_id[input_unit] = len(net_weights)
            # connect edge to the input node
            pins.append(input_id_to_vertex_id[input_unit])
            # connect edge to the work units
            for work_unit in work_units:
                pins.append(work_id_to_vertex_id[work_unit])
            xpins.append(len(pins))
            assert xpins[-1] - xpins[-2] > 1, "self edge"
            net_weights.append(
                _round_net_weight(workload.input_sizes[input_unit])
            )
        for output_unit, work_units in output_data_unit_to_work_units.items():
            output_id_to_net_id[output_unit] = len(net_weights)
            # connect edge to the output node
            pins.append(output_id_to_vertex_id[output_unit])
            # connect edge to the work units
            for work_unit in work_units:
                pins.append(work_id_to_vertex_id[work_unit])
            xpins.append(len(pins))
            assert xpins[-1] - xpins[-2] > 1, "self edge"
            net_weights.append(
                _round_net_weight(workload.output_sizes[output_unit])
            )
        # rescale weights to avoid overflow
        MAX_WEIGHT = 2**31 - 1 - 1000
        sum_net_weights = sum(net_weights)
        if sum_net_weights > MAX_WEIGHT:
            factor = MAX_WEIGHT / sum_net_weights
            net_weights = [max(int(w * factor), 1) for w in net_weights]
            weight_rounding_scale = weight_rounding_scale * factor
        sum_vertex_weights = sum(vertex_weights)
        actual_comp_epsilon = comp_epsilon * (
            inter_node_comp_epsilon_factor if current_node_id is None else 1
        )
        if (sum_vertex_weights * (1 + actual_comp_epsilon)) > MAX_WEIGHT:
            # if logging_enabled:
            #     logger.debug(
            #         "Rescaling due to weight overflow: before scaling vertex weights: {}".format(
            #             vertex_weights
            #         )
            #     )
            vertex_scale_factor = MAX_WEIGHT / (
                sum_vertex_weights * (1 + actual_comp_epsilon)
            )
            vertex_weights = [
                max(int(w * vertex_scale_factor), 1) for w in vertex_weights
            ]
            mem_weights = [
                max(int(w * vertex_scale_factor), 1) for w in mem_weights
            ]
            compute_weights = [
                max(int(w * vertex_scale_factor), 1) for w in compute_weights
            ]
        else:
            vertex_scale_factor = 1
        if logging_enabled:
            logger.debug(
                f"Constructed hypergraph with {len(pins)} nodes, "
                f"{len(net_weights) - 1} hyperedges."
            )
            # logger.debug(f"Edge weights: {net_weights}")
            # logger.debug(f"Node weights: {vertex_weights}")
            # logger.debug(f"Min node weight: {min(vertex_weights)}")
            # logger.debug(f"Max node weight: {max(vertex_weights)}")

        if solver in [HGPSolver.KAHYPAR, HGPSolver.MTKAHYPAR]:
            # make sure no vertex has weight 0
            vertex_weights = [max(w, 1) for w in vertex_weights]
            sum_compute_weight = sum(compute_weights)
            # rescale epsilon to work only on compute weight
            # first calculate a memory imbalance upper bound
            per_device_mem = [0 for _ in range(k)]
            per_device_count = [0 for _ in range(k)]
            for w in sorted(mem_weights, reverse=True):
                min_device = per_device_mem.index(min(per_device_mem))
                per_device_mem[min_device] += w
                per_device_count[min_device] += 1
            # # add back the sum of compute weight
            # for i in range(k):
            #     per_device_mem[i] += sum_compute_weight * per_device_count[i]
            max_mem_weight = max(per_device_mem)
            sum_vertex_weights = sum(vertex_weights)
            if logging_enabled:
                # if min(per_device_count) < 1:
                #     logger.warning(
                #         "Some devices have no memory weight assigned: {}".format(
                #             per_device_count
                #         )
                #     )
                logger.debug(
                    "Sum compute weight (after overflow scaling): {}".format(
                        sum_compute_weight
                    )
                )
                logger.debug(
                    f"Max memory weight (after overflow scaling): {max_mem_weight}"
                )
                logger.debug(
                    f"Perfectly_balanced_weight (after overflow scaling): {sum_vertex_weights / k}"
                )
                logger.debug(
                    f"Perfectly balanced compute weight (after overflow scaling): {sum_compute_weight / k}"
                )
            perfectly_balanced_weight = sum_vertex_weights / k
            perfectly_balanced_compute_weight = sum_compute_weight / k
            max_tolerated_compute_weight = (
                perfectly_balanced_compute_weight
                * (
                    1
                    + comp_epsilon
                    * (
                        inter_node_comp_epsilon_factor
                        if current_node_id is None
                        else 1
                    )
                )
            )
            if (
                max_tolerated_compute_weight
                - perfectly_balanced_compute_weight
                < 1
            ):
                max_tolerated_compute_weight = (
                    perfectly_balanced_compute_weight + 1
                )
            max_tolerated_actual_weight = (
                max_mem_weight + max_tolerated_compute_weight
            )
            scaled_epsilon = (
                max_tolerated_actual_weight - perfectly_balanced_weight
            ) / perfectly_balanced_weight
            if logging_enabled:
                logger.debug(
                    "max_tolerated_compute_weight: {}".format(
                        max_tolerated_compute_weight
                    )
                )
            # calculate individual partition weights
            per_device_weight = [
                int(per_device_mem[i] + max_tolerated_compute_weight)
                for i in range(k)
            ]
            if logging_enabled:
                logger.debug(
                    f"Per device mem weight (expected): {per_device_mem}"
                )
                sum_per_device_weight = sum(per_device_weight)
                logger.debug(
                    f"Per device weight (expected): {per_device_weight}, sum: {sum_per_device_weight}"
                )
                sum_vertex_weight = sum(vertex_weights)
                logger.debug(
                    f"Sum vertex weights (expected): {sum_vertex_weight}"
                )
                if sum_per_device_weight <= sum_vertex_weight:
                    logger.warning(
                        "Sum of per device weights ({}) is less than sum of vertex weights ({})!".format(
                            sum_per_device_weight, sum_vertex_weight
                        )
                    )
        elif solver == HGPSolver.PATOH:
            # PATOH does not support setting separate epsilon for different dims
            scaled_epsilon = mem_epsilon
            per_device_weight = []

        if logging_enabled:
            logger.debug(
                "==" * 10 + " Done constructing hypergraph " + "==" * 10
            )
        return (
            scaled_epsilon,
            per_device_weight,
            pins,
            xpins,
            net_weights,
            vertex_weights,
            work_id_to_vertex_id,
            input_id_to_vertex_id,
            output_id_to_vertex_id,
            input_id_to_net_id,
            output_id_to_net_id,
            memory_scale,
            vertex_scale_factor,
        )

    def _construct_comm_graph_for_balance(
        work_id_to_vertex_id: Dict[int, int],
        input_id_to_vertex_id: Dict[int, int],
        output_id_to_vertex_id: Dict[int, int],
        vertex_to_block_ids: List[int],
    ):
        # in the new communication graph:
        # vertex = input unit + output unit, weight = (lambda - 1) *
        # net = device

        # relabel vertex ids
        new_vertex_to_old_vertex = sorted(
            list(
                set(input_id_to_vertex_id.values()).union(
                    set(output_id_to_vertex_id.values())
                )
            )
        )
        old_vertex_to_new_vertex = {
            old_vertex_id: new_vertex_id
            for new_vertex_id, old_vertex_id in enumerate(
                new_vertex_to_old_vertex
            )
        }
        # build reverse mapping from vertex id to input and output id
        old_vertex_to_input_ids = defaultdict(list)
        old_vertex_to_output_ids = defaultdict(list)
        for input_id, old_vertex_id in input_id_to_vertex_id.items():
            old_vertex_to_input_ids[old_vertex_id].append(input_id)
        for output_id, old_vertex_id in output_id_to_vertex_id.items():
            old_vertex_to_output_ids[old_vertex_id].append(output_id)
        # map input / output id to required devices
        input_id_required_devices = defaultdict(set)
        output_id_required_devices = defaultdict(set)
        required_vertices_per_block = defaultdict(set)
        for work_id, input_ids in enumerate(workload.work_unit_input_map):
            work_block_id = vertex_to_block_ids[work_id_to_vertex_id[work_id]]
            for input_id in input_ids:
                input_id_required_devices[input_id].add(work_block_id)
                required_vertices_per_block[work_block_id].add(
                    input_id_to_vertex_id[input_id]
                )
        for work_id, output_ids in enumerate(workload.work_unit_output_map):
            work_block_id = vertex_to_block_ids[work_id_to_vertex_id[work_id]]
            for output_id in output_ids:
                output_id_required_devices[output_id].add(work_block_id)
                required_vertices_per_block[work_block_id].add(
                    output_id_to_vertex_id[output_id]
                )
        # count the communication volume associated with each old vertex
        old_vertex_mem = defaultdict(int)
        old_vertex_comm_volume = defaultdict(int)
        for old_vertex_id, input_ids in old_vertex_to_input_ids.items():
            for input_id in input_ids:
                old_vertex_mem[old_vertex_id] += _round_net_weight(
                    workload.input_sizes[input_id]
                )
                old_vertex_comm_volume[old_vertex_id] += _round_net_weight(
                    workload.input_sizes[input_id]
                ) * (len(input_id_required_devices[input_id]) - 1)
        for old_vertex_id, output_ids in old_vertex_to_output_ids.items():
            for output_id in output_ids:
                old_vertex_mem[old_vertex_id] += _round_net_weight(
                    workload.output_sizes[output_id]
                )
                old_vertex_comm_volume[old_vertex_id] += _round_net_weight(
                    workload.output_sizes[output_id]
                ) * (len(output_id_required_devices[output_id]) - 1)
        max_vertex_weights = max(old_vertex_comm_volume.values())
        # compose memory balance constraint with communication balance
        new_vertex_weights = [0 for _ in range(len(old_vertex_to_new_vertex))]
        for old_vertex_id, new_vertex_id in old_vertex_to_new_vertex.items():
            weight = (
                max_vertex_weights * old_vertex_mem[old_vertex_id]
                + old_vertex_comm_volume[old_vertex_id]
            )
            new_vertex_weights[new_vertex_id] = weight
        # scale epsilon to work only on communication weight
        k = len(required_vertices_per_block)
        perfectly_balanced_weight = sum(new_vertex_weights) / k
        perfectly_balanced_comm_weight = (
            sum(old_vertex_comm_volume.values()) / k
        )
        max_tolerated_actual_weight = (
            perfectly_balanced_weight
            + perfectly_balanced_comm_weight * comm_epsilon
        )
        scaled_epsilon = (
            max_tolerated_actual_weight - perfectly_balanced_weight
        ) / perfectly_balanced_weight
        new_vertex_weights = [int(x / 1000) for x in new_vertex_weights]
        # create new nets
        nets = [
            sorted(
                [
                    old_vertex_to_new_vertex[old_vertex]
                    for old_vertex in required_vertices_per_block[block_id]
                ]
            )
            for block_id in range(k)
        ]
        net_weights = [1 for _ in range(len(nets))]
        new_input_id_to_vertex_id = {
            input_id: old_vertex_to_new_vertex[old_vertex_id]
            for input_id, old_vertex_id in input_id_to_vertex_id.items()
        }
        new_output_id_to_vertex_id = {
            output_id: old_vertex_to_new_vertex[old_vertex_id]
            for output_id, old_vertex_id in output_id_to_vertex_id.items()
        }
        return (
            scaled_epsilon,
            nets,
            net_weights,
            new_vertex_weights,
            new_input_id_to_vertex_id,
            new_output_id_to_vertex_id,
        )

    device_to_workload_map = defaultdict(list)
    device_to_input_map = defaultdict(list)
    device_to_output_map = defaultdict(list)

    if n_total_devices == 1:
        # no partitioning needed
        device_to_workload_map[(0, 0)] = list(range(len(workload.workloads)))
        device_to_input_map[(0, 0)] = list(range(len(workload.input_sizes)))
        device_to_output_map[(0, 0)] = list(range(len(workload.output_sizes)))
        for input_id in range(len(workload.input_sizes)):
            workload.input_to_device_map[input_id] = (0, 0)
        for output_id in range(len(workload.output_sizes)):
            workload.output_to_device_map[output_id] = (0, 0)
        return (
            device_to_workload_map,
            device_to_input_map,
            device_to_output_map,
        )

    inter_node_cost = 0
    intra_node_cost = 0

    def _print_input_id(input_id: int):
        meta = workload.block_mapping.input_id_to_meta[input_id]
        return "S{}H{}{}{}".format(
            meta.seq_id,
            meta.head_id,
            "Q" if meta.type == BlockType.Q else "KV",
            meta.block_id,
        )

    def _print_output_id(output_id: int):
        meta = workload.block_mapping.output_id_to_meta[output_id]
        return "S{}H{}O{}".format(meta.seq_id, meta.head_id, meta.block_id)

    def _hg_assign_nodes_by_heuristic(
        work_id_to_vertex_id: Dict[int, int],
        input_id_to_vertex_id: Dict[int, int],
        output_id_to_vertex_id: Dict[int, int],
    ):
        (
            device_to_workload_map,
            device_to_input_map,
            device_to_output_map,
        ) = assign_device_by_heuristic(context, workload)
        work_id_to_devce = {
            work_id: device
            for device, work_ids in device_to_workload_map.items()
            for work_id in work_ids
        }
        input_id_to_device = {
            input_id: device
            for device, input_ids in device_to_input_map.items()
            for input_id in input_ids
        }
        output_id_to_device = {
            output_id: device
            for device, output_ids in device_to_output_map.items()
            for output_id in output_ids
        }
        # assign the initial partitioning to the hypergraph
        initial_partition_per_vertex = [
            None for _ in range(len(vertex_weights))
        ]
        # block_ids = [None for _ in range(len(vertex_weights))]
        for work_id, vertex_id in work_id_to_vertex_id.items():
            initial_partition_per_vertex[vertex_id] = work_id_to_devce[
                work_id
            ][0]
            # block_ids[vertex_id] = work_id_to_devce[work_id][0]
        for input_id, vertex_id in input_id_to_vertex_id.items():
            initial_partition_per_vertex[vertex_id] = input_id_to_device[
                input_id
            ][0]
            # logger.debug("Input {}, vertex {}, assigned to node {}".format(_print_input_id(input_id), vertex_id, input_id_to_device[input_id][0]))
            # block_ids[vertex_id] = input_id_to_device[input_id][0]
        for output_id, vertex_id in output_id_to_vertex_id.items():
            initial_partition_per_vertex[vertex_id] = output_id_to_device[
                output_id
            ][0]
            # block_ids[vertex_id] = output_id_to_device[output_id][0]
            # logger.debug("Output {}, vertex {}, assigned to node {}".format(_print_output_id(output_id), vertex_id, output_id_to_device[output_id][0]))
        for x in initial_partition_per_vertex:
            assert x is not None and x < context.n_nodes
        return initial_partition_per_vertex

    def _improve_balance_local_search(
        work_id_to_node_id: Dict[int, int],
        input_id_to_node_id: Dict[int, int],
        output_id_to_node_id: Dict[int, int],
    ):
        # repeat:
        # 1. find a node N with highest egress traffic
        # 2. find the input id with highest number of required nodes
        # 3. for each required node N', try to move all work units requiring
        #    this input to N (which reduces egress traffic from N)
        #    at the same time, find a set of work units to move from the chosen
        #    node to the required node (that will not increase current egress
        #    traffic)
        #    calculate the new highest egress traffic, if it's reduced, accept
        #    the move
        pass

    if context.n_nodes > 1:
        assert (
            n_total_devices % context.n_nodes == 0
        ), "n_devices must be a multiple of n_nodes"
        # multi-node setup, do two-level hierarchical partitioning
        (
            scaled_epsilon,
            per_device_weight,
            pins,
            xpins,
            net_weights,
            vertex_weights,
            work_id_to_vertex_id,
            input_id_to_vertex_id,
            output_id_to_vertex_id,
            _,
            _,
            mem_scale,
            vertex_scale_factor,
        ) = _construct_graph_for_work_ids(
            context.n_nodes, range(len(workload.workloads))
        )
        if solver == HGPSolver.KAHYPAR:
            hg = kahypar.Hypergraph(
                len(vertex_weights),
                len(net_weights),
                xpins,
                pins,
                context.n_nodes,
                net_weights,
                vertex_weights,
            )
            kahypar_context = kahypar.Context()
            kahypar_context.loadINIconfiguration(
                os.path.join(
                    os.path.dirname(__file__), "config", "kahypar_config.ini"
                )
            )
            kahypar_context.setCustomTargetBlockWeights(per_device_weight)
            kahypar_context.setK(context.n_nodes)
            kahypar_context.setEpsilon(scaled_epsilon)
            if improve_upon_heuristic:
                initial_partition_per_vertex = _hg_assign_nodes_by_heuristic(
                    work_id_to_vertex_id,
                    input_id_to_vertex_id,
                    output_id_to_vertex_id,
                )
                kahypar.improve(
                    hg,
                    initial_partition_per_vertex,
                    num_vcycles,
                    kahypar_context,
                )
            else:
                kahypar.partition(hg, kahypar_context)
            block_ids = [hg.blockID(i) for i in range(len(vertex_weights))]
            inter_node_cost = (
                kahypar.connectivityMinusOne(hg) / weight_rounding_scale
            )
            if logging_enabled:
                logger.debug(
                    "Inter-node cost: {:.4f} MB".format(inter_node_cost / 1e6)
                )
                logger.debug(
                    f"Imbalance: {kahypar.imbalance(hg, kahypar_context)}, constriant: {scaled_epsilon}"
                )
                logger.debug(
                    "Per block weight: {}".format(
                        [hg.blockWeight(i) for i in range(context.n_nodes)]
                    )
                )
                # log memory weight per node
                per_node_mem = [0 for _ in range(context.n_nodes)]
                per_node_compute = [0 for _ in range(context.n_nodes)]
                for input_id, vertex_id in input_id_to_vertex_id.items():
                    per_node_mem[block_ids[vertex_id]] += (
                        workload.input_sizes[input_id]
                        * mem_scale
                        * vertex_scale_factor
                    )
                    # logger.debug(f"Input {input_id} (vertex {vertex_id}) of size {workload.input_sizes[input_id] / 1e6}M, vertex weight: {vertex_weights[vertex_id]}, init on node: {block_ids[vertex_id]}")
                for output_id, vertex_id in output_id_to_vertex_id.items():
                    per_node_mem[block_ids[vertex_id]] += (
                        workload.output_sizes[output_id]
                        * mem_scale
                        * vertex_scale_factor
                    )
                    # logger.debug(f"Output {output_id} (vertex {vertex_id}) of size {workload.output_sizes[output_id] / 1e6}M, vertex weight: {vertex_weights[vertex_id]}, init on node: {block_ids[vertex_id]}")
                for work_id, vertex_id in work_id_to_vertex_id.items():
                    per_node_compute[block_ids[vertex_id]] += (
                        workload.workloads[work_id]
                        * workload_scale
                        * vertex_scale_factor
                    )
                for node_id, mem in enumerate(per_node_mem):
                    logger.debug(
                        f"Node {node_id} memory weight: {mem}, compute weight: {per_node_compute[node_id]}"
                    )

            # total_comm = 0
            # for input_id in range(len(workload.input_sizes)):
            #     related_work_ids = []
            #     for work_id in range(len(workload.workloads)):
            #         if input_id in workload.work_unit_input_map[work_id]:
            #             related_work_ids.append(work_id)
            #     related_nodes = set([block_ids[work_id_to_vertex_id[work_id]] for work_id in related_work_ids])
            #     indicent_nodes = related_nodes.copy()
            #     indicent_nodes.add(block_ids[input_id_to_vertex_id[input_id]])
            #     required_comm = (len(indicent_nodes) - 1) * workload.input_sizes[input_id]
            #     total_comm += required_comm
            #     logger.debug("Input {} of size {}M, init on node: {}, required by nodes: {}, comm: {}M".format(_print_input_id(input_id), workload.input_sizes[input_id] / 1e6,
            #         block_ids[input_id_to_vertex_id[input_id]], related_nodes, required_comm / 1e6))
            # for output_id in range(len(workload.output_sizes)):
            #     related_work_ids = []
            #     for work_id in range(len(workload.workloads)):
            #         if output_id in workload.work_unit_output_map[work_id]:
            #             related_work_ids.append(work_id)
            #     related_nodes = set([block_ids[work_id_to_vertex_id[work_id]] for work_id in related_work_ids])
            #     indicent_nodes = related_nodes.copy()
            #     indicent_nodes.add(block_ids[output_id_to_vertex_id[output_id]])
            #     required_comm = (len(indicent_nodes) - 1) * workload.output_sizes[output_id]
            #     total_comm += required_comm
            #     logger.debug("Output {} of size {}M, init on node: {}, required by nodes: {}, comm: {}M".format(_print_output_id(output_id), workload.output_sizes[output_id] / 1e6,
            #         block_ids[output_id_to_vertex_id[output_id]], related_nodes, required_comm / 1e6))
            # print("Total comm: {}M".format(total_comm / 1e6))
            assert not enforce_comm_balance
        elif solver == HGPSolver.MTKAHYPAR:
            # we always use single thread partitioning since it's parallelized
            # across iterations
            mtk = mtkahypar.initialize(1)
            mtk_context = mtk.context_from_preset(mtkahypar.PresetType.QUALITY)
            mtk_context.set_partitioning_parameters(
                context.n_nodes, scaled_epsilon, mtkahypar.Objective.KM1
            )
            mtkahypar.set_seed(42)
            mtk_context.logging = False
            # mtk uses nested list hyperedge input
            mtk_hypedges = [
                pins[xpins[i] : xpins[i + 1]] for i in range(len(xpins) - 1)
            ]
            hg = mtk.create_hypergraph(
                mtk_context,
                len(vertex_weights),
                len(net_weights),
                mtk_hypedges,
                vertex_weights,
                net_weights,
            )
            if improve_upon_heuristic:
                initial_partition_per_vertex = _hg_assign_nodes_by_heuristic(
                    work_id_to_vertex_id,
                    input_id_to_vertex_id,
                    output_id_to_vertex_id,
                )
                part_hg = hg.create_partitioned_hypergraph(
                    context.n_nodes,
                    mtk_context,
                    initial_partition_per_vertex,
                )
                part_hg.improve_partition(mtk_context, num_vcycles)
            else:
                part_hg = hg.partition(mtk_context)
            block_ids = [
                part_hg.block_id(i) for i in range(len(vertex_weights))
            ]
            inter_node_cost = part_hg.km1() * weight_rounding_scale
            if logging_enabled:
                logger.debug(f"Inter-node cost: {inter_node_cost}")
                logger.debug(
                    f"Imbalance: {part_hg.imbalance(mtk_context)}, constriant: {scaled_epsilon}"
                )
        elif solver == HGPSolver.PATOH:
            if improve_upon_heuristic:
                raise ValueError(
                    "Improve upon heuristic is not supported for PATOH"
                )
            block_ids, _, inter_node_cost = pypatoh.part_cmd(
                context.n_nodes,
                N_CONSTRS,
                vertex_weights,
                net_weights,
                xpins,
                pins,
                epsilon=scaled_epsilon,
                mode=mode,
            )
            if logging_enabled:
                logger.debug(
                    "Inter-node cost: {:.4f} MB".format(
                        inter_node_cost / weight_rounding_scale / 1e6
                    )
                )
                # log memory weight per node
                per_node_mem = [0 for _ in range(context.n_nodes)]
                per_node_compute = [0 for _ in range(context.n_nodes)]
                for input_id, vertex_id in input_id_to_vertex_id.items():
                    per_node_mem[block_ids[vertex_id]] += (
                        workload.input_sizes[input_id]
                        * mem_scale
                        * vertex_scale_factor
                    )
                    # logger.debug(f"Input {input_id} (vertex {vertex_id}) of size {workload.input_sizes[input_id] / 1e6}M, vertex weight: {vertex_weights[vertex_id]}, init on node: {block_ids[vertex_id]}")
                for output_id, vertex_id in output_id_to_vertex_id.items():
                    per_node_mem[block_ids[vertex_id]] += (
                        workload.output_sizes[output_id]
                        * mem_scale
                        * vertex_scale_factor
                    )
                    # logger.debug(f"Output {output_id} (vertex {vertex_id}) of size {workload.output_sizes[output_id] / 1e6}M, vertex weight: {vertex_weights[vertex_id]}, init on node: {block_ids[vertex_id]}")
                for work_id, vertex_id in work_id_to_vertex_id.items():
                    per_node_compute[block_ids[vertex_id]] += (
                        workload.workloads[work_id]
                        * workload_scale
                        * vertex_scale_factor
                    )
                for node_id, mem in enumerate(per_node_mem):
                    logger.debug(
                        f"Node {node_id} memory weight: {mem}, compute weight: {per_node_compute[node_id]}"
                    )
        # get the partitioning of units to nodes
        work_id_to_node_id = {
            work_id: block_ids[vertex_id]
            for work_id, vertex_id in work_id_to_vertex_id.items()
        }
        if enforce_comm_balance:
            raise NotImplementedError("Communication balance is not supported")
            logger.debug("Before _construct_comm_graph_for_balance")
            (
                scaled_comm_epsilon,
                nets,
                net_weights,
                new_vertex_weights,
                new_input_id_to_vertex_id,
                new_output_id_to_vertex_id,
            ) = _construct_comm_graph_for_balance(
                work_id_to_vertex_id,
                input_id_to_vertex_id,
                output_id_to_vertex_id,
                block_ids,
            )
            logger.debug("After _construct_comm_graph_for_balance")
            mtk_context = mtk.context_from_preset(mtkahypar.PresetType.QUALITY)
            logger.debug("After create another mtk context")
            mtk_context.set_partitioning_parameters(
                context.n_nodes, scaled_comm_epsilon, mtkahypar.Objective.KM1
            )
            logger.debug("After set_partitioning_parameters")
            mtkahypar.set_seed(42)
            mtk_context.logging = True
            logger.debug("Before create_hypergraph")
            comm_hg = mtk.create_hypergraph(
                mtk_context,
                len(new_vertex_weights),
                len(net_weights),
                nets,
                new_vertex_weights,
                net_weights,
            )
            logger.debug("After create_hypergraph")
            part_comm_hg = comm_hg.partition(mtk_context)
            comm_block_ids = [
                part_comm_hg.block_id(i)
                for i in range(len(new_vertex_weights))
            ]
            input_id_to_node_id = {
                input_id: comm_block_ids[vertex_id]
                for input_id, vertex_id in new_input_id_to_vertex_id.items()
            }
            output_id_to_node_id = {
                output_id: comm_block_ids[vertex_id]
                for output_id, vertex_id in new_output_id_to_vertex_id.items()
            }
        else:
            input_id_to_node_id = {
                input_id: block_ids[vertex_id]
                for input_id, vertex_id in input_id_to_vertex_id.items()
            }
            output_id_to_node_id = {
                output_id: block_ids[vertex_id]
                for output_id, vertex_id in output_id_to_vertex_id.items()
            }
        # print i/o required nodes
        # for input_id, node_id in input_id_to_node_id.items():
        #     required_nodes = set()
        #     for work_id, input_ids in enumerate(workload.work_unit_input_map):
        #         if input_id in input_ids:
        #             required_nodes.add(work_id_to_node_id[work_id])
        #     logger.debug(
        #         f"Input {_print_input_id(input_id)} assigned to node {node_id}, required by nodes {sorted(required_nodes)}"
        #     )
        # for output_id, node_id in output_id_to_node_id.items():
        #     required_nodes = set()
        #     for work_id, output_ids in enumerate(
        #         workload.work_unit_output_map
        #     ):
        #         if output_id in output_ids:
        #             required_nodes.add(work_id_to_node_id[work_id])
        #     logger.debug(
        #         f"Output {_print_output_id(output_id)} assigned to node {node_id}, required by nodes {sorted(required_nodes)}"
        #     )

        for input_id in range(len(workload.input_sizes)):
            if input_id not in input_id_to_node_id:
                raise ValueError(
                    f"Input {input_id} is not assigned to any node"
                )
        for output_id in range(len(workload.output_sizes)):
            if output_id not in output_id_to_node_id:
                raise ValueError(
                    f"Output {output_id} is not assigned to any node"
                )

        workload_per_node = defaultdict(list)
        for work_id, node_id in work_id_to_node_id.items():
            workload_per_node[node_id].append(work_id)
        for work_id in range(len(workload.workloads)):
            if work_id not in work_id_to_node_id:
                raise ValueError(
                    f"Workload {work_id} is not assigned to any node"
                )
        # build per node hypergraphs
        n_devices_per_node = context.n_devices_per_node
        for node_id, work_unit_ids in workload_per_node.items():
            if n_devices_per_node > 1:
                if logging_enabled:
                    logger.debug(
                        "==" * 10
                        + f"Performing intra-node partitioning for node {node_id}"
                        + "==" * 10
                    )
                (
                    scaled_epsilon,
                    intra_node_per_device_weight,
                    intra_node_pins,
                    intra_node_xpins,
                    intra_node_net_weights,
                    intra_node_vertex_weights,
                    intra_node_work_id_to_vertex_id,
                    intra_node_input_id_to_vertex_id,
                    intra_node_output_id_to_vertex_id,
                    _,
                    _,
                    intra_node_mem_scale,
                    intra_node_vertex_scale_factor,
                ) = _construct_graph_for_work_ids(
                    n_devices_per_node,
                    work_unit_ids,
                    current_node_id=node_id,
                    input_to_node_map=input_id_to_node_id,
                    output_to_node_map=output_id_to_node_id,
                )
                if solver == HGPSolver.KAHYPAR:
                    hg = kahypar.Hypergraph(
                        len(intra_node_vertex_weights),
                        len(intra_node_net_weights),
                        intra_node_xpins,
                        intra_node_pins,
                        n_devices_per_node,
                        intra_node_net_weights,
                        intra_node_vertex_weights,
                    )
                    kahypar_context = kahypar.Context()
                    kahypar_context.loadINIconfiguration(
                        os.path.join(
                            os.path.dirname(__file__),
                            "config",
                            "kahypar_config.ini",
                        )
                    )
                    kahypar_context.setCustomTargetBlockWeights(
                        intra_node_per_device_weight
                    )
                    kahypar_context.setK(n_devices_per_node)
                    kahypar_context.setEpsilon(scaled_epsilon)
                    kahypar.partition(hg, kahypar_context)
                    intra_node_block_ids = [
                        hg.blockID(i)
                        for i in range(len(intra_node_vertex_weights))
                    ]
                    per_node_intra_node_cost = (
                        kahypar.connectivityMinusOne(hg)
                        / weight_rounding_scale
                    )
                    if logging_enabled:
                        logger.debug(
                            f"Intra-node cost: {per_node_intra_node_cost / 1e6} MB"
                        )
                        logger.debug(
                            f"Imbalance: {kahypar.imbalance(hg, kahypar_context)}, constriant: {scaled_epsilon}"
                        )
                    if logging_enabled:
                        # log memory weight per device
                        per_device_mem = [
                            0 for _ in range(context.n_devices_per_node)
                        ]
                        per_device_compute = [
                            0 for _ in range(context.n_devices_per_node)
                        ]
                        for (
                            input_id,
                            vertex_id,
                        ) in intra_node_input_id_to_vertex_id.items():
                            if input_id_to_node_id[input_id] == node_id:
                                per_device_mem[
                                    intra_node_block_ids[vertex_id]
                                ] += (
                                    workload.input_sizes[input_id]
                                    * intra_node_mem_scale
                                    * intra_node_vertex_scale_factor
                                )
                        for (
                            output_id,
                            vertex_id,
                        ) in intra_node_output_id_to_vertex_id.items():
                            if output_id_to_node_id[output_id] == node_id:
                                per_device_mem[
                                    intra_node_block_ids[vertex_id]
                                ] += (
                                    workload.output_sizes[output_id]
                                    * intra_node_mem_scale
                                    * intra_node_vertex_scale_factor
                                )
                        for (
                            work_id,
                            vertex_id,
                        ) in intra_node_work_id_to_vertex_id.items():
                            per_device_compute[
                                intra_node_block_ids[vertex_id]
                            ] += (
                                workload.workloads[work_id]
                                * workload_scale
                                * intra_node_vertex_scale_factor
                            )
                        block_weights = [
                            hg.blockWeight(i)
                            for i in range(n_devices_per_node)
                        ]
                        for dev_id, mem in enumerate(per_device_mem):
                            logger.debug(
                                f"Device {dev_id} memory weight: {mem}, compute weight: {per_device_compute[dev_id]}, block weight: {block_weights[dev_id]}"
                            )
                elif solver == HGPSolver.MTKAHYPAR:
                    mtk_context = mtk.context_from_preset(
                        mtkahypar.PresetType.QUALITY
                    )
                    mtk_context.set_partitioning_parameters(
                        n_devices_per_node,
                        scaled_epsilon,
                        mtkahypar.Objective.KM1,
                    )
                    mtkahypar.set_seed(42)
                    mtk_context.logging = False
                    mtk_hypedges = [
                        intra_node_pins[
                            intra_node_xpins[i] : intra_node_xpins[i + 1]
                        ]
                        for i in range(len(intra_node_xpins) - 1)
                    ]
                    hg = mtk.create_hypergraph(
                        mtk_context,
                        len(intra_node_vertex_weights),
                        len(intra_node_net_weights),
                        mtk_hypedges,
                        intra_node_vertex_weights,
                        intra_node_net_weights,
                    )
                    part_hg = hg.partition(mtk_context)
                    intra_node_block_ids = [
                        part_hg.block_id(i)
                        for i in range(len(intra_node_vertex_weights))
                    ]
                    per_node_intra_node_cost = part_hg.km1()
                elif solver == HGPSolver.PATOH:
                    intra_node_block_ids, _, per_node_intra_node_cost = (
                        pypatoh.part_cmd(
                            n_devices_per_node,
                            N_CONSTRS,
                            intra_node_vertex_weights,
                            intra_node_net_weights,
                            intra_node_xpins,
                            intra_node_pins,
                            epsilon=scaled_epsilon,
                            mode=mode,
                        )
                    )
                # get the partitioning of units to nodes
                work_id_to_device_id = {
                    work_id: intra_node_block_ids[vertex_id]
                    for work_id, vertex_id in intra_node_work_id_to_vertex_id.items()
                }
                input_id_to_device_id = {
                    input_id: intra_node_block_ids[vertex_id]
                    for input_id, vertex_id in intra_node_input_id_to_vertex_id.items()
                }
                output_id_to_device_id = {
                    output_id: intra_node_block_ids[vertex_id]
                    for output_id, vertex_id in intra_node_output_id_to_vertex_id.items()
                }
            else:
                # only one device per node, partition algo seems to have some
                # issues with this case, manually assign all workloads to device 0
                work_id_to_device_id = {
                    work_id: 0 for work_id in work_unit_ids
                }
                input_id_to_device_id = {
                    input_id: 0
                    for input_id in range(len(workload.input_sizes))
                }
                output_id_to_device_id = {
                    output_id: 0
                    for output_id in range(len(workload.output_sizes))
                }
                per_node_intra_node_cost = 0
            # fill the device map
            for work_id, device_id in work_id_to_device_id.items():
                device_to_workload_map[(node_id, device_id)].append(work_id)
            # only assign device id if the input/output is initially assigned
            # to the same node
            for input_id, device_id in input_id_to_device_id.items():
                if input_id_to_node_id[input_id] == node_id:
                    device_to_input_map[(node_id, device_id)].append(input_id)
            for output_id, device_id in output_id_to_device_id.items():
                if output_id_to_node_id[output_id] == node_id:
                    device_to_output_map[(node_id, device_id)].append(
                        output_id
                    )
            intra_node_cost += per_node_intra_node_cost
        # it is possible that a input/output is assiged to a node but only
        # required by workloads in another node, which is not captured in the
        # intra-node partitioning process.
        # Manually assign them to the device with the lowest memory footprint
        assigned_inputs = set()
        assigned_outputs = set()
        manual_assignment_counter = 0
        for inputs in device_to_input_map.values():
            assigned_inputs.update(inputs)
        for outputs in device_to_output_map.values():
            assigned_outputs.update(outputs)
        device_to_memory_map = defaultdict(int)
        for d in device_to_input_map.keys():
            device_to_memory_map[d] = sum(
                [workload.input_sizes[x] for x in device_to_input_map[d]]
            ) + sum(
                [workload.output_sizes[x] for x in device_to_output_map[d]]
            )
        node_manual_assignment_count = defaultdict(int)
        for input_id, node_id in input_id_to_node_id.items():
            if input_id not in assigned_inputs:
                # manual assignment
                # first check colocation constraints
                colocated_inputs = set([input_id])
                colocated_outputs = set()
                for (
                    co_input_ids,
                    co_output_ids,
                ) in workload.colocation_constraints:
                    if input_id in co_input_ids:
                        # check all constrained input/outputs are unassigned
                        for co_input_id in co_input_ids:
                            assert (
                                co_input_id not in assigned_inputs
                            ), f"Input {co_input_id} is already assigned"
                            assert input_id_to_node_id[co_input_id] == node_id
                        for co_output_id in co_output_ids:
                            assert (
                                co_output_id not in assigned_outputs
                            ), f"Output {co_output_id} is already assigned"
                            assert (
                                output_id_to_node_id[co_output_id] == node_id
                            )
                        colocated_inputs = colocated_inputs.union(co_input_ids)
                        colocated_outputs = colocated_outputs.union(
                            co_output_ids
                        )
                manual_assignment_counter += 1
                min_device = None
                min_memory = float("inf")
                for device_id in range(n_devices_per_node):
                    if device_to_memory_map[(node_id, device_id)] < min_memory:
                        min_memory = device_to_memory_map[(node_id, device_id)]
                        min_device = device_id
                for co_input_id in colocated_inputs:
                    device_to_input_map[(node_id, min_device)].append(
                        co_input_id
                    )
                    assigned_inputs.add(co_input_id)
                    device_to_memory_map[
                        (node_id, min_device)
                    ] += workload.input_sizes[co_input_id]
                for co_output_id in colocated_outputs:
                    device_to_output_map[(node_id, min_device)].append(
                        co_output_id
                    )
                    assigned_outputs.add(co_output_id)
                    device_to_memory_map[
                        (node_id, min_device)
                    ] += workload.output_sizes[co_output_id]
                node_manual_assignment_count[node_id] += 1
        for output_id, node_id in output_id_to_node_id.items():
            if output_id not in assigned_outputs:
                # manual assignment
                manual_assignment_counter += 1
                # first check colocation constraints
                colocated_inputs = set()
                colocated_outputs = set([output_id])
                for (
                    co_input_ids,
                    co_output_ids,
                ) in workload.colocation_constraints:
                    if output_id in co_output_ids:
                        # check all constrained input/outputs are unassigned
                        for co_input_id in co_input_ids:
                            assert (
                                co_input_id not in assigned_inputs
                            ), f"Input {co_input_id} is already assigned"
                            assert input_id_to_node_id[co_input_id] == node_id
                        for co_output_id in co_output_ids:
                            assert (
                                co_output_id not in assigned_outputs
                            ), f"Output {co_output_id} is already assigned"
                            assert (
                                output_id_to_node_id[co_output_id] == node_id
                            )
                        colocated_inputs = colocated_inputs.union(co_input_ids)
                        colocated_outputs = colocated_outputs.union(
                            co_output_ids
                        )
                min_device = None
                min_memory = float("inf")
                for device_id in range(n_devices_per_node):
                    if device_to_memory_map[(node_id, device_id)] < min_memory:
                        min_memory = device_to_memory_map[(node_id, device_id)]
                        min_device = device_id
                for co_input_id in colocated_inputs:
                    device_to_input_map[(node_id, min_device)].append(
                        co_input_id
                    )
                    assigned_inputs.add(co_input_id)
                    device_to_memory_map[
                        (node_id, min_device)
                    ] += workload.input_sizes[co_input_id]
                for co_output_id in colocated_outputs:
                    device_to_output_map[(node_id, min_device)].append(
                        co_output_id
                    )
                    assigned_outputs.add(co_output_id)
                    device_to_memory_map[
                        (node_id, min_device)
                    ] += workload.output_sizes[co_output_id]
                node_manual_assignment_count[node_id] += 1
        if logging_enabled:
            logger.debug(
                "Manually assigned {} inputs/outputs to devices.".format(
                    manual_assignment_counter
                )
            )
            for node_id, count in node_manual_assignment_count.items():
                logger.debug(
                    "Node {}: assigned {} inputs/outputs.".format(
                        node_id, count
                    )
                )
    else:
        # direct partition
        (
            scaled_epsilon,
            per_device_weight,
            pins,
            xpins,
            net_weights,
            vertex_weights,
            work_id_to_vertex_id,
            input_id_to_vertex_id,
            output_id_to_vertex_id,
            _,
            _,
            _,
            _,
        ) = _construct_graph_for_work_ids(
            n_total_devices, range(len(workload.workloads))
        )
        if solver == HGPSolver.KAHYPAR:
            hg = kahypar.Hypergraph(
                len(vertex_weights),
                len(net_weights),
                xpins,
                pins,
                n_total_devices,
                net_weights,
                vertex_weights,
            )
            kahypar_context = kahypar.Context()
            kahypar_context.loadINIconfiguration(
                os.path.join(
                    os.path.dirname(__file__), "config", "kahypar_config.ini"
                )
            )
            kahypar_context.setCustomTargetBlockWeights(per_device_weight)
            kahypar_context.setK(n_total_devices)
            kahypar_context.setEpsilon(scaled_epsilon)
            kahypar.partition(hg, kahypar_context)
            block_ids = [hg.blockID(i) for i in range(len(vertex_weights))]
            intra_node_cost = kahypar.connectivityMinusOne(hg)
        elif solver == HGPSolver.MTKAHYPAR:
            mtk = mtkahypar.initialize(1)
            mtk_context = mtk.context_from_preset(mtkahypar.PresetType.QUALITY)
            mtk_context.set_partitioning_parameters(
                n_total_devices,
                scaled_epsilon,
                mtkahypar.Objective.KM1,
            )
            mtkahypar.set_seed(42)
            mtk_context.logging = False
            mtk_hypedges = [
                pins[xpins[i] : xpins[i + 1]] for i in range(len(xpins) - 1)
            ]
            hg = mtk.create_hypergraph(
                mtk_context,
                len(vertex_weights),
                len(net_weights),
                mtk_hypedges,
                vertex_weights,
                net_weights,
            )
            part_hg = hg.partition(mtk_context)
            block_ids = [
                part_hg.block_id(i) for i in range(len(vertex_weights))
            ]
            intra_node_cost = part_hg.km1()
        elif solver == HGPSolver.PATOH:
            block_ids, block_weight, intra_node_cost = pypatoh.part_cmd(
                n_total_devices,
                N_CONSTRS,
                vertex_weights,
                net_weights,
                xpins,
                pins,
                epsilon=mem_epsilon,
                mode=mode,
            )
        # get the partitioning of units
        node_id = 0
        for work_id, vertex_id in work_id_to_vertex_id.items():
            device_id = block_ids[vertex_id]
            device_to_workload_map[(node_id, device_id)].append(work_id)
        for input_id, vertex_id in input_id_to_vertex_id.items():
            device_id = block_ids[vertex_id]
            device_to_input_map[(node_id, device_id)].append(input_id)
        for output_id, vertex_id in output_id_to_vertex_id.items():
            device_id = block_ids[vertex_id]
            device_to_output_map[(node_id, device_id)].append(output_id)

    if logging_enabled:
        for d, work_ids in device_to_workload_map.items():
            logger.debug(
                f"Device {d}: total compute weight {sum([workload.workloads[x] for x in work_ids])}, "
                f"total memory weight {sum([workload.input_sizes[x] for x in device_to_input_map[d]]) + sum([workload.output_sizes[x] for x in device_to_output_map[d]])}"
            )

    for d, w_ids in device_to_workload_map.items():
        for w_id in w_ids:
            workload.work_to_device_map[w_id] = d
    for d, i_ids in device_to_input_map.items():
        for i_id in i_ids:
            workload.input_to_device_map[i_id] = d
    for d, o_ids in device_to_output_map.items():
        for o_id in o_ids:
            workload.output_to_device_map[o_id] = d

    return (
        device_to_workload_map,
        device_to_input_map,
        device_to_output_map,
    )


def assign_input_outputs(
    input_sizes: List[int],
    output_sizes: List[int],
    work_unit_input_map: List[List[int]],
    work_unit_output_map: List[List[int]],
    device_to_workload_map: Dict[Tuple[int, int], List[int]],
    colocated_input_outputs: Optional[List[List[List[int]]]] = None,
    logger: Optional[logging.Logger] = None,
    input_to_name: Optional[Dict[int, str]] = None,
    output_to_name: Optional[Dict[int, str]] = None,
):
    input_to_device_map = {}
    output_to_device_map = {}

    # first generate list of candidate devices for each input and output
    input_to_candidate_devices = defaultdict(set)
    output_to_candidate_devices = defaultdict(set)
    for d, workloads in device_to_workload_map.items():
        for w in workloads:
            for input_id in work_unit_input_map[w]:
                input_to_candidate_devices[input_id].add(d)
            for output_id in work_unit_output_map[w]:
                output_to_candidate_devices[output_id].add(d)
    if input_to_name is not None and logger is not None:
        logger.debug("Input to candidate devices:")
        for input_id, candidate_devices in input_to_candidate_devices.items():
            logger.debug(f"\t{input_to_name[input_id]}: {candidate_devices}")
    if output_to_name is not None and logger is not None:
        logger.debug("Output to candidate devices:")
        for (
            output_id,
            candidate_devices,
        ) in output_to_candidate_devices.items():
            logger.debug(f"\t{output_to_name[output_id]}: {candidate_devices}")
    # if input output colocation constraints are provided, merge the candidate
    # devices for colocated data
    colocated_input_output_to_candidate_devices = []
    free_input_to_candidate_devices = defaultdict(list)
    free_output_to_candidate_devices = defaultdict(list)
    input_ids_in_constraints = set()
    output_ids_in_constraints = set()
    if colocated_input_outputs:
        for input_ids, output_ids in colocated_input_outputs:
            candidate_devices = None
            for input_id in input_ids:
                if candidate_devices is None:
                    candidate_devices = set(
                        input_to_candidate_devices[input_id]
                    )
                else:
                    candidate_devices &= set(
                        input_to_candidate_devices[input_id]
                    )
            for output_id in output_ids:
                if candidate_devices is None:
                    candidate_devices = set(
                        output_to_candidate_devices[output_id]
                    )
                else:
                    candidate_devices &= set(
                        output_to_candidate_devices[output_id]
                    )
            colocated_input_output_to_candidate_devices.append(
                (input_ids, output_ids, candidate_devices)
            )
            input_ids_in_constraints.update(input_ids)
            output_ids_in_constraints.update(output_ids)
    for input_id in input_to_candidate_devices:
        if input_id not in input_ids_in_constraints:
            free_input_to_candidate_devices[input_id] = (
                input_to_candidate_devices[input_id]
            )
    for output_id in output_to_candidate_devices:
        if output_id not in output_ids_in_constraints:
            free_output_to_candidate_devices[output_id] = (
                output_to_candidate_devices[output_id]
            )
    if logger is not None:
        logger.debug("colocated_input_output_to_candidate_devices:")
        for (
            input_ids,
            output_ids,
            candidate_devices,
        ) in colocated_input_output_to_candidate_devices:
            logger.debug(
                f"\t{[input_to_name[x] for x in input_ids]} -> {[output_to_name[x] for x in output_ids]}: {candidate_devices}"
            )
    # main consideration is memory balance, therefore we greedily assign inputs
    # and outputs to least loaded devices
    device_loads = defaultdict(float)
    # first consider the colocated inputs and outputs
    for (
        input_ids,
        output_ids,
        candidate_devices,
    ) in colocated_input_output_to_candidate_devices:
        # get the least loaded device
        least_loaded_device = min(
            candidate_devices, key=lambda d: device_loads[d]
        )
        for input_id in input_ids:
            input_to_device_map[input_id] = least_loaded_device
            device_loads[least_loaded_device] += input_sizes[input_id]
        for output_id in output_ids:
            output_to_device_map[output_id] = least_loaded_device
            device_loads[least_loaded_device] += output_sizes[output_id]
    # now assign the free inputs and outputs
    for input_id, candidate_devices in free_input_to_candidate_devices.items():
        least_loaded_device = min(
            candidate_devices, key=lambda d: device_loads[d]
        )
        input_to_device_map[input_id] = least_loaded_device
        device_loads[least_loaded_device] += input_sizes[input_id]
    for (
        output_id,
        candidate_devices,
    ) in free_output_to_candidate_devices.items():
        least_loaded_device = min(
            candidate_devices, key=lambda d: device_loads[d]
        )
        output_to_device_map[output_id] = least_loaded_device
        device_loads[least_loaded_device] += output_sizes[output_id]
    if logger is not None:
        logger.debug("Device loads:")
        for d, load in device_loads.items():
            logger.debug(f"\tDevice {d}: {load}")
    return input_to_device_map, output_to_device_map


def assign_device_by_heuristic(
    context: ExecutionContext,
    workload: WorkloadSpec,
):
    n_devices = context.n_devices_per_node * context.n_nodes
    n_heads = len(
        set(
            m.head_id for m in workload.block_mapping.input_id_to_meta.values()
        )
    )
    a2a_degree = min(n_heads, n_devices)
    assert n_devices % a2a_degree == 0
    assert n_heads % a2a_degree == 0
    p2p_degree = n_devices // a2a_degree
    # separate workloads by seq_id and head_id
    per_seq_head_inputs = defaultdict(lambda: defaultdict(list))
    per_seq_head_outputs = defaultdict(lambda: defaultdict(list))
    per_seq_head_workloads = defaultdict(lambda: defaultdict(list))
    for input_id in workload.block_mapping.input_id_to_meta:
        meta = workload.block_mapping.input_id_to_meta[input_id]
        per_seq_head_inputs[meta.seq_id][meta.head_id].append(input_id)
    for output_id in workload.block_mapping.output_id_to_meta:
        meta = workload.block_mapping.output_id_to_meta[output_id]
        per_seq_head_outputs[meta.seq_id][meta.head_id].append(output_id)
    for work_id in range(len(workload.workloads)):
        meta = workload.block_mapping.work_id_to_meta[work_id]
        per_seq_head_workloads[meta.seq_id][meta.head_id].append(work_id)
    # sort by block id
    for seq_id in per_seq_head_inputs.keys():
        for head_id in per_seq_head_inputs[seq_id].keys():
            per_seq_head_inputs[seq_id][head_id].sort(
                key=lambda x: workload.block_mapping.input_id_to_meta[
                    x
                ].block_id
            )
            per_seq_head_outputs[seq_id][head_id].sort(
                key=lambda x: workload.block_mapping.output_id_to_meta[
                    x
                ].block_id
            )
    # def _print_input_id(input_id: int):
    #     meta = workload.block_mapping.input_id_to_meta[input_id]
    #     return "S{}H{}{}{}".format(
    #         meta.seq_id,
    #         meta.head_id,
    #         "Q" if meta.type == BlockType.Q else "KV",
    #         meta.block_id,
    #     )

    # def _print_output_id(output_id: int):
    #     meta = workload.block_mapping.output_id_to_meta[output_id]
    #     return "S{}H{}O{}".format(meta.seq_id, meta.head_id, meta.block_id)

    # assign workload, inputs and outputs
    for seq_id in sorted(list(per_seq_head_inputs.keys())):
        n_blocks = len(per_seq_head_inputs[seq_id][0]) // 2
        n_blocks_per_p2p_dev = (n_blocks + p2p_degree - 1) // p2p_degree
        block_id_to_p2p_device = {}
        remaining_blocks = list(range(n_blocks))
        for device_id in range(p2p_degree):
            remaining_blocks_for_curr_device = n_blocks_per_p2p_dev
            front = True
            while remaining_blocks_for_curr_device > 0 and remaining_blocks:
                if front:
                    block_id = remaining_blocks.pop(0)
                else:
                    block_id = remaining_blocks.pop()
                block_id_to_p2p_device[block_id] = device_id
                remaining_blocks_for_curr_device -= 1
                front = not front
        block_id_to_all_device = {}
        remaining_blocks = list(range(n_blocks))
        for device_id in range(n_devices):
            remaining_blocks_for_curr_device = (
                n_blocks + n_devices - 1
            ) // n_devices
            front = True
            while remaining_blocks_for_curr_device > 0 and remaining_blocks:
                if front:
                    block_id = remaining_blocks.pop(0)
                else:
                    block_id = remaining_blocks.pop()
                block_id_to_all_device[block_id] = device_id
                remaining_blocks_for_curr_device -= 1
                front = not front
        n_heads_per_a2a_dev = n_heads // a2a_degree
        for head_id in sorted(list(per_seq_head_inputs[seq_id].keys())):
            for workload_id in per_seq_head_workloads[seq_id][head_id]:
                output_id = workload.work_unit_output_map[workload_id][0]
                a2a_device_id = head_id // n_heads_per_a2a_dev
                p2p_device_id = block_id_to_p2p_device[
                    workload.block_mapping.output_id_to_meta[
                        output_id
                    ].block_id
                ]
                actual_device_id = p2p_device_id * a2a_degree + a2a_device_id
                actual_node_id = actual_device_id // context.n_devices_per_node
                actual_device_id_in_node = (
                    actual_device_id % context.n_devices_per_node
                )
                device = (actual_node_id, actual_device_id_in_node)
                workload.work_to_device_map[workload_id] = device
            for input_id in per_seq_head_inputs[seq_id][head_id]:
                block_id = workload.block_mapping.input_id_to_meta[
                    input_id
                ].block_id
                actual_device_id = block_id_to_all_device[block_id]
                actual_node_id = actual_device_id // context.n_devices_per_node
                actual_device_id_in_node = (
                    actual_device_id % context.n_devices_per_node
                )
                device = (actual_node_id, actual_device_id_in_node)
                workload.input_to_device_map[input_id] = device
            for output_id in per_seq_head_outputs[seq_id][head_id]:
                block_id = workload.block_mapping.output_id_to_meta[
                    output_id
                ].block_id
                actual_device_id = block_id_to_all_device[block_id]
                actual_node_id = actual_device_id // context.n_devices_per_node
                actual_device_id_in_node = (
                    actual_device_id % context.n_devices_per_node
                )
                device = (actual_node_id, actual_device_id_in_node)
                workload.output_to_device_map[output_id] = device
    device_to_input_map = defaultdict(list)
    for input_id, d in workload.input_to_device_map.items():
        device_to_input_map[d].append(input_id)
    device_to_output_map = defaultdict(list)
    for output_id, d in workload.output_to_device_map.items():
        device_to_output_map[d].append(output_id)
    device_to_workload_map = defaultdict(list)
    for workload_id, d in workload.work_to_device_map.items():
        device_to_workload_map[d].append(workload_id)
    return (
        device_to_workload_map,
        device_to_input_map,
        device_to_output_map,
    )
