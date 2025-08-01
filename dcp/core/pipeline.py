import copy
import time
import json
import logging
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

from dcp.core.block_table import DataBlockMeta, BlockType
from dcp.core.steiner_tree_packing import pack_multicast_trees
from dcp.utils.logger import read_env_bool
from dcp_cpp import greedy_selection_per_device

PIPELINE_LOGGING_ENABLED = read_env_bool("DCP_LOG_PIPELINE", default=False)
PIPELINE_DISABLE_TWO_PHASE_COMM = read_env_bool(
    "DCP_DISABLE_TWO_PHASE_COMM", default=False
)


# def greedy_selection_per_device(
#     all_devices: Set[Tuple[int, int]],
#     scheduled_workloads: set[int],
#     workload_costs: List[float],
#     input_sizes: List[float],
#     workload_input_map: List[List[int]],
#     input_to_meta_map: Dict[int, DataBlockMeta],
#     device_to_workload_map: Dict[Tuple[int, int], List[int]],
#     device_to_local_inputs_map: Dict[Tuple[int, int], Dict[int, int]],
#     target_comm_load_per_node: Dict[Tuple[int, int], float],
#     stage_id: int,
#     logger: Optional[logging.Logger] = None,
# ):
#     """
#     Greedily select a set of workloads and communication so that:
#     1) Total communication time is less or equal to target_comm_time
#     2) Maximize the ready computation time on all devices
#     """
#     all_devices = set(all_devices)
#     unlocked_workloads_per_device = defaultdict(list)
#     # dst_d -> [(input_id, src_d)]
#     performed_comm_per_device = defaultdict(list)

#     current_ready_workload_time_per_device = defaultdict(float)
#     current_cross_node_comm_load_per_device = defaultdict(float)
#     current_intra_node_comm_load_per_device = defaultdict(float)

#     def _check_comm_constraints(input_ids, dst_node, dst_device):
#         # input_id -> src_d
#         internode_required_comms = {}
#         intranode_required_comms = {}
#         internode_comm_required_sizes = defaultdict(float)
#         intranode_comm_required_sizes = defaultdict(float)
#         for input_id in input_ids:
#             # first check if the input is local
#             if input_id in device_to_local_inputs_map[(dst_node, dst_device)]:
#                 continue
#             # requires communication
#             # first try to find a local device that has the input
#             local_devices_having_input = set()
#             for (
#                 d,
#                 local_inputs_and_stages,
#             ) in device_to_local_inputs_map.items():
#                 if (
#                     d[0] == dst_node
#                     and input_id in local_inputs_and_stages
#                     and local_inputs_and_stages[input_id] < stage_id
#                 ):
#                     local_devices_having_input.add(d)
#             if local_devices_having_input:
#                 # choose one with the minimum communication load
#                 min_device = min(
#                     local_devices_having_input,
#                     key=lambda d: current_intra_node_comm_load_per_device[d],
#                 )
#                 intranode_required_comms[input_id] = min_device
#                 intranode_comm_required_sizes[min_device] += input_sizes[
#                     input_id
#                 ]
#                 continue
#             # requires inter-node communication
#             cross_node_devices_having_input = set()
#             for (
#                 d,
#                 local_inputs_and_stages,
#             ) in device_to_local_inputs_map.items():
#                 if (
#                     d[0] != dst_node
#                     and input_id in local_inputs_and_stages
#                     and local_inputs_and_stages[input_id] < stage_id
#                 ):
#                     cross_node_devices_having_input.add(d)
#             if not cross_node_devices_having_input:
#                 logger.debug(
#                     f"Stage: {stage_id}, DstD: ({dst_node}, {dst_device}), No device has input {input_id}"
#                 )
#                 for (
#                     d,
#                     local_inputs_and_stages,
#                 ) in device_to_local_inputs_map.items():
#                     logger.debug(f"Device {d}: {local_inputs_and_stages}")
#                 assert False
#             # choose one with the minimum communication load
#             min_device = min(
#                 cross_node_devices_having_input,
#                 key=lambda d: current_cross_node_comm_load_per_device[d],
#             )
#             internode_required_comms[input_id] = min_device
#             internode_comm_required_sizes[min_device] += input_sizes[input_id]
#         for d, comm_load in internode_comm_required_sizes.items():
#             if (
#                 current_cross_node_comm_load_per_device[d] + comm_load
#                 > target_comm_load_per_node[d]
#             ):
#                 return False, None, None, None, None
#         # for d, comm_load in intranode_comms_for_input_ids.items():
#         #     if (
#         #         current_intra_node_comm_load_per_device[d] + comm_load
#         #         > target_comm_load_per_device[d]
#         #     ):
#         #         return False, None, None
#         # check required comm size matches required comm
#         inter_req_size = defaultdict(float)
#         intra_req_size = defaultdict(float)
#         for input_id, src_d in internode_required_comms.items():
#             inter_req_size[src_d] += input_sizes[input_id]
#         for src_d, comm_size in inter_req_size.items():
#             assert abs(comm_size - internode_comm_required_sizes[src_d]) < 1e-6
#         for input_id, src_d in intranode_required_comms.items():
#             intra_req_size[src_d] += input_sizes[input_id]
#         for src_d, comm_size in intra_req_size.items():
#             assert abs(comm_size - intranode_comm_required_sizes[src_d]) < 1e-6
#         return (
#             True,
#             internode_required_comms,
#             intranode_required_comms,
#             internode_comm_required_sizes,
#             intranode_comm_required_sizes,
#         )

#     devices_to_skip = set()
#     while True:
#         # choose the device with the minimum ready computation time
#         (current_node, current_device) = min(
#             [d for d in all_devices if d not in devices_to_skip],
#             key=lambda d: current_ready_workload_time_per_device[d],
#         )
#         pending_workloads = sorted(
#             [
#                 w
#                 for w in device_to_workload_map[(current_node, current_device)]
#                 if w not in scheduled_workloads
#                 and w
#                 not in unlocked_workloads_per_device[
#                     (current_node, current_device)
#                 ]
#             ],
#             key=lambda w: workload_costs[w],
#             reverse=True,
#         )
#         if not pending_workloads:
#             if logger is not None and PIPELINE_LOGGING_ENABLED:
#                 logger.debug(
#                     "No workloads to schedule for device ({}, {})".format(
#                         current_node, current_device
#                     )
#                 )
#             devices_to_skip.add((current_node, current_device))
#             if devices_to_skip == all_devices:
#                 break
#             else:
#                 if logger is not None and PIPELINE_LOGGING_ENABLED:
#                     logger.debug("Devices to skip: {}".format(devices_to_skip))
#                     logger.debug("All devices: {}".format(all_devices))
#                 assert len(devices_to_skip) < len(all_devices)
#             continue
#         # if logger is not None:
#         #     logger.debug(
#         #         "Selecting workloads for device ({}, {}), pending_workloads={}".format(
#         #             current_node, current_device, pending_workloads
#         #         )
#         #     )
#         # greedily select the workloads that will not violate communication
#         # constraints
#         progress = False
#         for w in pending_workloads:
#             # preprocess inputs to make sure out and lse are scheduled together
#             workload_inputs = []
#             output_id = None
#             lse_id = None
#             for input_id in workload_input_map[w]:
#                 input_meta = input_to_meta_map[input_id]
#                 if input_meta.type == BlockType.Out:
#                     output_id = input_id
#                 elif input_meta.type == BlockType.LSE:
#                     lse_id = input_id
#                 else:
#                     workload_inputs.append(input_id)
#             assert (output_id is None) == (lse_id is None)
#             if output_id is not None:
#                 workload_inputs.append(output_id)
#             (
#                 satisfies_comm_constraints,
#                 internode_required_comms,
#                 intranode_required_comms,
#                 inter_node_comm_loads,
#                 intra_node_comm_loads,
#             ) = _check_comm_constraints(
#                 workload_inputs, current_node, current_device
#             )
#             if satisfies_comm_constraints:
#                 # schedule the workload
#                 unlocked_workloads_per_device[
#                     (current_node, current_device)
#                 ].append(w)
#                 # update communication loads
#                 for input_id, src_d in internode_required_comms.items():
#                     performed_comm_per_device[
#                         (current_node, current_device)
#                     ].append((input_id, src_d))
#                     if input_id == output_id:
#                         performed_comm_per_device[
#                             (current_node, current_device)
#                         ].append((lse_id, src_d))
#                 for input_id, src_d in intranode_required_comms.items():
#                     performed_comm_per_device[
#                         (current_node, current_device)
#                     ].append((input_id, src_d))
#                     if input_id == output_id:
#                         performed_comm_per_device[
#                             (current_node, current_device)
#                         ].append((lse_id, src_d))
#                 for src_d, comm_load in inter_node_comm_loads.items():
#                     current_cross_node_comm_load_per_device[src_d] += comm_load
#                 for src_d, comm_load in intra_node_comm_loads.items():
#                     current_intra_node_comm_load_per_device[src_d] += comm_load
#                 for input_id in workload_input_map[w]:
#                     if (
#                         input_id
#                         not in device_to_local_inputs_map[
#                             (current_node, current_device)
#                         ]
#                     ):
#                         device_to_local_inputs_map[
#                             (current_node, current_device)
#                         ][input_id] = stage_id
#                 # update ready workload time
#                 current_ready_workload_time_per_device[
#                     (current_node, current_device)
#                 ] += workload_costs[w]
#                 progress = True
#                 break
#         if not progress:
#             break
#     # logger.debug("Before squeezing in additional communication")
#     # for d, comms in performed_comm_per_device.items():
#     #     logger.debug(
#     #         "Device {}: comms={}, local inputs: {}".format(d, comms, device_to_local_inputs_map[d])
#     #     )
#     # next, squeeze in addtional communication if possible
#     remaining_workloads_per_device = defaultdict(list)
#     for current_node, current_device in all_devices:
#         pending_workloads = sorted(
#             [
#                 w
#                 for w in device_to_workload_map[(current_node, current_device)]
#                 if w not in scheduled_workloads
#                 and w
#                 not in unlocked_workloads_per_device[
#                     (current_node, current_device)
#                 ]
#             ],
#             key=lambda w: workload_costs[w],
#             reverse=True,
#         )
#         remaining_workloads_per_device[(current_node, current_device)] = (
#             pending_workloads
#         )

#     for d in all_devices:
#         # find a communication that can be scheduled
#         for w in remaining_workloads_per_device[d]:
#             output_id = None
#             lse_id = None
#             for input_id in workload_input_map[w]:
#                 input_meta = input_to_meta_map[input_id]
#                 if input_meta.type == BlockType.Out:
#                     output_id = input_id
#                 elif input_meta.type == BlockType.LSE:
#                     lse_id = input_id
#             assert (output_id is None) == (lse_id is None)
#             for input_id in workload_input_map[w]:
#                 if input_id == lse_id:
#                     continue
#                 if input_id not in device_to_local_inputs_map[d]:
#                     (
#                         satisfies_comm_constraints,
#                         internode_required_comms,
#                         intranode_required_comms,
#                         inter_node_comm_loads,
#                         intra_node_comm_loads,
#                     ) = _check_comm_constraints([input_id], d[0], d[1])
#                     if satisfies_comm_constraints:
#                         # schedule the communication
#                         for src_d, comm_load in inter_node_comm_loads.items():
#                             performed_comm_per_device[d].append(
#                                 (input_id, src_d)
#                             )
#                             if input_id == output_id:
#                                 performed_comm_per_device[d].append(
#                                     (lse_id, src_d)
#                                 )
#                         for src_d, comm_load in intra_node_comm_loads.items():
#                             performed_comm_per_device[d].append(
#                                 (input_id, src_d)
#                             )
#                             if input_id == output_id:
#                                 performed_comm_per_device[d].append(
#                                     (lse_id, src_d)
#                                 )
#                         for src_d, comm_load in inter_node_comm_loads.items():
#                             current_cross_node_comm_load_per_device[
#                                 src_d
#                             ] += comm_load
#                         for src_d, comm_load in intra_node_comm_loads.items():
#                             current_intra_node_comm_load_per_device[
#                                 src_d
#                             ] += comm_load
#                         device_to_local_inputs_map[d][input_id] = stage_id
#                         if input_id == output_id:
#                             device_to_local_inputs_map[d][lse_id] = stage_id
#     if logger is not None:
#         for d in set(current_intra_node_comm_load_per_device.keys()).union(
#             current_cross_node_comm_load_per_device.keys()
#         ):
#             if PIPELINE_LOGGING_ENABLED:
#                 logger.debug(
#                     "Current communication load for device {}: intra_node={} (target: {}), inter_node={} (target: {})".format(
#                         d,
#                         current_intra_node_comm_load_per_device[d],
#                         0,
#                         # target_comm_load_per_device[d],
#                         current_cross_node_comm_load_per_device[d],
#                         target_comm_load_per_node[d],
#                     )
#                 )
#     return unlocked_workloads_per_device, performed_comm_per_device


def check_and_visualize_pipeline(
    workloads: List[int],
    input_sizes: List[int],
    work_unit_input_map: List[List[int]],
    workloads_per_stage: List[Dict[Tuple[int, int], List[int]]],
    input_to_device_map: Dict[int, Tuple[int, int]],
    inter_node_bandwidth: float,
    intra_node_bandwidth: float,
    out_file: str = "pipeline.json",
):
    per_stage_comp_per_device = []
    per_stage_comm_per_device = []
    current_on_device_inputs = defaultdict(list)
    for input, d in input_to_device_map.items():
        current_on_device_inputs[d].append(input)
    for stage_id, stage in enumerate(workloads_per_stage):
        curr_stage_comp_per_device = defaultdict(float)
        curr_stage_total_comm_per_device = defaultdict(float)
        curr_stage_inter_send_per_device = defaultdict(float)
        curr_stage_inter_recv_per_device = defaultdict(float)
        curr_stage_intra_send_per_device = defaultdict(float)
        curr_stage_intra_recv_per_device = defaultdict(float)
        for d, per_stage_workloads in stage.items():
            for w in per_stage_workloads:
                # check if inputs are available
                for input in work_unit_input_map[w]:
                    if input not in current_on_device_inputs[d]:
                        # communication
                        curr_node_id, curr_device_id = d
                        src_node_id, src_device_id = input_to_device_map[input]
                        assert stage_id > 0
                        intra_node = curr_node_id == src_node_id
                        if intra_node:
                            curr_stage_intra_send_per_device[
                                (src_node_id, src_device_id)
                            ] += (input_sizes[input] / intra_node_bandwidth)
                            curr_stage_intra_recv_per_device[
                                (curr_node_id, curr_device_id)
                            ] += (input_sizes[input] / intra_node_bandwidth)
                        else:
                            curr_stage_inter_send_per_device[
                                (src_node_id, src_device_id)
                            ] += (input_sizes[input] / inter_node_bandwidth)
                            curr_stage_inter_recv_per_device[
                                (curr_node_id, curr_device_id)
                            ] += (input_sizes[input] / inter_node_bandwidth)
                        current_on_device_inputs[d].append(input)
                # computation
                curr_stage_comp_per_device[d] += workloads[w]
            total_comm_time = max(
                curr_stage_inter_send_per_device[d],
                curr_stage_inter_recv_per_device[d],
                curr_stage_intra_send_per_device[d],
                curr_stage_intra_recv_per_device[d],
            )
            curr_stage_total_comm_per_device[d] = total_comm_time
        per_stage_comp_per_device.append(curr_stage_comp_per_device)
        per_stage_comm_per_device.append(curr_stage_total_comm_per_device)
    result_json = []
    curr_time = 0
    for stage_id in range(len(per_stage_comp_per_device)):
        stage_time = 0
        for (node_id, device_id), comp_time in per_stage_comp_per_device[
            stage_id
        ].items():
            result_json.append(
                {
                    "name": f"Stage {stage_id} Compute",
                    "ph": "X",  # Complete Event (Begin + End event)
                    "cat": "comp",
                    "ts": curr_time,
                    "dur": comp_time,
                    "tid": "Compute",
                    "pid": f"Device ({node_id}, {device_id})",
                    "cname": "thread_state_iowait",
                }
            )
            stage_time = max(stage_time, comp_time)
        if stage_id < len(per_stage_comp_per_device) - 1:
            for (node_id, device_id), comm_time in per_stage_comm_per_device[
                stage_id + 1
            ].items():
                result_json.append(
                    {
                        "name": f"Stage {stage_id + 1} Communication",
                        "ph": "X",  # Complete Event (Begin + End event)
                        "cat": "comm",
                        "ts": curr_time,
                        "dur": comm_time,
                        "tid": "Communication",
                        "pid": f"Device ({node_id}, {device_id})",
                        "cname": "thread_state_running",
                    }
                )
                stage_time = max(stage_time, comm_time)
        # add instant event
        result_json.append(
            {
                "name": f"Stage {stage_id}",
                "ph": "i",  # Instant Event
                "ts": curr_time + stage_time,
                "s": "g",
            }
        )
        curr_time += stage_time

    with open(out_file, "w") as f:
        json.dump(result_json, f)


def log_pipeline_comm_cost(
    comms_per_stage_per_device: List[
        Dict[Tuple[int, int], List[Tuple[int, Tuple[int, int]]]]
    ],
    input_sizes: List[float],
    logger: logging.Logger,
):
    # calculate cross node comm volume per stage
    actual_internode_send_volume_per_node = [
        defaultdict(float) for _ in range(len(comms_per_stage_per_device))
    ]
    actual_internode_recv_volume_per_node = [
        defaultdict(float) for _ in range(len(comms_per_stage_per_device))
    ]
    for stage_id in range(len(comms_per_stage_per_device)):
        for d, comms in comms_per_stage_per_device[stage_id].items():
            for input_id, src_d in comms:
                src_node_id, src_device_id = src_d
                dst_node_id, dst_device_id = d
                if src_node_id != dst_node_id:
                    actual_internode_send_volume_per_node[stage_id][
                        src_node_id
                    ] += input_sizes[input_id]
                    actual_internode_recv_volume_per_node[stage_id][
                        dst_node_id
                    ] += input_sizes[input_id]
    total_vol = 0
    for stage_id in range(len(comms_per_stage_per_device)):
        max_send_vol = 0
        max_recv_vol = 0
        nodes = set(
            actual_internode_send_volume_per_node[stage_id].keys()
        ).union(actual_internode_recv_volume_per_node[stage_id].keys())
        for d in nodes:
            max_send_vol = max(
                max_send_vol,
                actual_internode_send_volume_per_node[stage_id][d],
            )
            max_recv_vol = max(
                max_recv_vol,
                actual_internode_recv_volume_per_node[stage_id][d],
            )
        logger.debug(
            "Stage {}: max_send={}, max_recv={}, max_vol={}".format(
                stage_id,
                max_send_vol,
                max_recv_vol,
                max(max_send_vol, max_recv_vol),
            )
        )
        total_vol += max(max_send_vol, max_recv_vol)
    logger.debug("Max total volume per node: {}".format(total_vol))


def generate_pipelines_n_stages(
    n_stages: int,
    all_devices: Set[Tuple[int, int]],
    workload_costs: List[float],
    workload_input_map: List[List[int]],
    input_to_meta_map: Dict[int, DataBlockMeta],
    input_sizes: List[float],
    workload_to_device_map: Dict[int, Tuple[int, int]],
    input_to_device_map: Dict[int, Tuple[int, int]],
    is_bw: bool,
    logger: Optional[logging.Logger] = None,
):
    """
    Given the workload assignment, input and output relations, group
    the workloads and communication into pipelines with n stages
    """
    # first calculate the total communication needed for each device / node
    total_internode_comm_volume_per_node = defaultdict(float)
    total_intranode_comm_volume_per_device = defaultdict(float)

    device_to_workload_map: Dict[Tuple[int, int], List] = {}
    for d in all_devices:
        device_to_workload_map[d] = []
    for workload_id, d in workload_to_device_map.items():
        device_to_workload_map[d].append(workload_id)

    # keep track of which inputs are on the current device, and in which stage
    # is the input obtained
    device_to_local_inputs_map: Dict[Tuple[int, int], Dict] = {}
    for d in all_devices:
        device_to_local_inputs_map[d] = {}
    for input_id, d in input_to_device_map.items():
        device_to_local_inputs_map[d][input_id] = -1

    curr_inputs_per_device = defaultdict(set)
    for input_id, (node_id, device_id) in input_to_device_map.items():
        curr_inputs_per_device[(node_id, device_id)].add(input_id)

    for workload_id, (
        work_node_id,
        work_device_id,
    ) in workload_to_device_map.items():
        for input_id in workload_input_map[workload_id]:
            if (
                input_id
                in curr_inputs_per_device[(work_node_id, work_device_id)]
            ):
                continue
            src_node_id, src_device_id = input_to_device_map[input_id]
            if src_node_id == work_node_id:
                # intra-node communication
                total_intranode_comm_volume_per_device[
                    (src_node_id, src_device_id)
                ] += input_sizes[input_id]
            else:
                # inter-node communication
                total_internode_comm_volume_per_node[
                    (src_node_id, src_device_id)
                ] += input_sizes[input_id]
                # suppose the input is immediately broadcast to all local devices
                for d in all_devices:
                    if d[0] == work_node_id:
                        total_intranode_comm_volume_per_device[
                            d
                        ] += input_sizes[input_id]
                        curr_inputs_per_device[d].add(input_id)
            curr_inputs_per_device[(work_node_id, work_device_id)].add(
                input_id
            )
    if logger is not None and PIPELINE_LOGGING_ENABLED:
        for d in all_devices:
            logger.debug(
                "Node {}: total estimated internode comm volume={}".format(
                    d, total_internode_comm_volume_per_node[d]
                )
            )
            logger.debug(
                "Device {}: total estimated intranode comm volume={}".format(
                    d, total_intranode_comm_volume_per_device[d]
                )
            )
    total_internode_volume_across_all_devices = 0
    for d, total_volume in total_internode_comm_volume_per_node.items():
        total_internode_volume_across_all_devices += total_volume
    target_volume_per_device_per_stage = {}
    for d in all_devices:
        target_volume_per_device_per_stage[d] = (
            total_internode_volume_across_all_devices
            / len(all_devices)
            / n_stages
        ) * 1.2
    # now greedily build stages following:
    # 1. select workloads to be performed and schedule the required communication
    #    such that the total communication volume is less than the target
    # 2. schedule additional communication to balance communication volume
    workload_per_stage_per_device = []
    # dst_d -> [(input_id, src_d)]
    communication_per_stage_per_device = []

    scheduled_workloads = set()

    ready_workloads_per_device = defaultdict(list)
    # first, mark all local computation blocks that can be done without
    # any communication as ready and schedule them as the first comp stage
    for workload_id in range(len(workload_costs)):
        node_id, device_id = workload_to_device_map[workload_id]
        input_all_local = all(
            input_to_device_map[input_id] == (node_id, device_id)
            for input_id in workload_input_map[workload_id]
        )
        if input_all_local:
            ready_workloads_per_device[(node_id, device_id)].append(
                workload_id
            )

    if logger is not None and PIPELINE_LOGGING_ENABLED:
        logger.debug("Stage 0:")
        for d, workloads in ready_workloads_per_device.items():
            logger.debug(
                "\tdevice=({}), initially ready workload time={}".format(
                    d, sum(workload_costs[w] for w in workloads)
                )
            )

    input_to_block_type_map = [None] * len(input_sizes)
    for input_id in range(len(input_sizes)):
        input_to_block_type_map[input_id] = input_to_meta_map[input_id].type

    first_stage_comp_workloads = ready_workloads_per_device
    workload_per_stage_per_device.append(first_stage_comp_workloads)
    communication_per_stage_per_device.append(defaultdict(list))
    # update scheduled workloads
    for d, workloads in first_stage_comp_workloads.items():
        scheduled_workloads.update(workloads)

    for stage_id in range(n_stages):
        if logger is not None and PIPELINE_LOGGING_ENABLED:
            logger.debug("Before scheduling stage {}".format(stage_id))
            for d, local_inputs in device_to_local_inputs_map.items():
                logger.debug(
                    "\tD({}), local inputs={}".format(d, local_inputs)
                )
        unlocked_workloads, required_comms, device_to_local_inputs_map = (
            greedy_selection_per_device(
                set(all_devices),
                scheduled_workloads,
                workload_costs,
                input_sizes,
                workload_input_map,
                input_to_block_type_map,
                device_to_workload_map,
                device_to_local_inputs_map,
                target_volume_per_device_per_stage,
                is_bw,
                stage_id,
                logger if PIPELINE_LOGGING_ENABLED else None,
            )
        )
        # unlocked_workloads, required_comms = greedy_selection_per_device(
        #     all_devices,
        #     scheduled_workloads,
        #     workload_costs,
        #     input_sizes,
        #     workload_input_map,
        #     input_to_meta_map,
        #     device_to_workload_map,
        #     device_to_local_inputs_map,
        #     target_volume_per_device_per_stage,
        #     stage_id,
        #     logger,
        # )
        workload_for_this_stage = unlocked_workloads
        if logger is not None and PIPELINE_LOGGING_ENABLED:
            for d, workloads in workload_for_this_stage.items():
                logger.debug(
                    "\tD({}), stage compute_time={}".format(
                        d, sum(workload_costs[w] for w in workloads)
                    )
                )
        # update scheduled workloads
        for d, workloads in workload_for_this_stage.items():
            scheduled_workloads.update(workloads)
        if stage_id == n_stages - 1:
            if logger is not None and PIPELINE_LOGGING_ENABLED:
                logger.debug("Scheduling last stage comp.")
            # last stage, schedule any remaining workloads
            target_volume_per_device_last_stage = {
                d: float("inf") for d in all_devices
            }
            (
                unlocked_workloads,
                additional_required_comms,
                device_to_local_inputs_map,
            ) = greedy_selection_per_device(
                set(all_devices),
                scheduled_workloads,
                workload_costs,
                input_sizes,
                workload_input_map,
                input_to_block_type_map,
                device_to_workload_map,
                device_to_local_inputs_map,
                target_volume_per_device_last_stage,
                is_bw,
                stage_id,
                logger if PIPELINE_LOGGING_ENABLED else None,
            )
            # unlocked_workloads, additional_required_comms = (
            #     greedy_selection_per_device(
            #         all_devices,
            #         scheduled_workloads,
            #         workload_costs,
            #         input_sizes,
            #         workload_input_map,
            #         input_to_meta_map,
            #         device_to_workload_map,
            #         device_to_local_inputs_map,
            #         target_volume_per_device_last_stage,
            #         stage_id,
            #         logger,
            #     )
            # )
            for d, workloads in unlocked_workloads.items():
                if d not in workload_for_this_stage:
                    workload_for_this_stage[d] = set()
                workload_for_this_stage[d] = workload_for_this_stage[d].union(
                    workloads
                )
                # workload_for_this_stage[d].extend(workloads)
                scheduled_workloads.update(workloads)
            # check
            remaining_workloads = {
                d: [
                    w
                    for w in device_to_workload_map[d]
                    if w not in scheduled_workloads
                ]
                for d in all_devices
            }
            assert not any(remaining_workloads.values())
            for dst_d, comm in additional_required_comms.items():
                if dst_d not in required_comms:
                    required_comms[dst_d] = []
                required_comms[dst_d].extend(comm)
        # add the unlocked workloads to the pipeline
        workload_per_stage_per_device.append(workload_for_this_stage)
        communication_per_stage_per_device.append(required_comms)
    # check that all workloads are scheduled
    actual_scheduled_workloads = set()
    for stage_workload_dict in workload_per_stage_per_device:
        for d, workloads in stage_workload_dict.items():
            actual_scheduled_workloads.update(workloads)
    assert actual_scheduled_workloads == set(range(len(workload_costs)))
    # do a post processing to separate out second phase comms
    if PIPELINE_DISABLE_TWO_PHASE_COMM:
        postprocessed_comm_per_stage_per_device = (
            communication_per_stage_per_device
        )
        second_phase_comm_per_stage_per_device = [defaultdict(list)] * len(
            communication_per_stage_per_device
        )
    else:
        postprocessed_comm_per_stage_per_device = [defaultdict(list)]
        second_phase_comm_per_stage_per_device = [defaultdict(list)]
        for stage_id, comms_per_device in enumerate(
            communication_per_stage_per_device[1:]
        ):
            total_internode_recv_per_stage = defaultdict(float)
            dst_devices_per_input = defaultdict(set)
            for dst_d, comms in comms_per_device.items():
                for input_id, src_d in comms:
                    if src_d[0] != dst_d[0]:
                        total_internode_recv_per_stage[dst_d] += input_sizes[
                            input_id
                        ]
                        dst_devices_per_input[(src_d, input_id, dst_d[0])].add(
                            dst_d[1]
                        )
            sorted_dst_devices_per_input_keys = sorted(
                list(dst_devices_per_input.keys()),
                key=lambda x: x[1],
            )
            comms_to_remove = defaultdict(list)
            second_phase_comms = defaultdict(list)
            for (
                src_d,
                input_id,
                dst_node,
            ) in sorted_dst_devices_per_input_keys:
                dst_devices = dst_devices_per_input[
                    (src_d, input_id, dst_node)
                ]
                if len(dst_devices) > 1:
                    # split the communication into two phases
                    # choose the device with the minimum recv load as internode
                    # recver
                    min_recv_device = min(
                        dst_devices,
                        key=lambda d: total_internode_recv_per_stage[
                            (dst_node, d)
                        ],
                    )
                    # remove other communications
                    for d in dst_devices:
                        if d != min_recv_device:
                            comms_to_remove[(dst_node, d)].append(
                                (input_id, src_d)
                            )
                            second_phase_comms[(dst_node, d)].append(
                                (input_id, (dst_node, min_recv_device))
                            )
            new_comms_per_device = defaultdict(list)
            for dst_d, comms in comms_per_device.items():
                new_comms = []
                for comm in comms:
                    if comm not in comms_to_remove[dst_d]:
                        new_comms.append(comm)
                new_comms_per_device[dst_d] = new_comms
            postprocessed_comm_per_stage_per_device.append(
                new_comms_per_device
            )
            second_phase_comm_per_stage_per_device.append(second_phase_comms)
    return (
        workload_per_stage_per_device,
        postprocessed_comm_per_stage_per_device,
        second_phase_comm_per_stage_per_device,
    )


def generate_pipelines_n_stages_with_tree_packing(
    n_stages: int,
    all_devices: Set[Tuple[int, int]],
    workload_costs: List[float],
    workload_input_map: List[List[int]],
    input_to_meta_map: Dict[int, DataBlockMeta],
    input_sizes: List[float],
    workload_to_device_map: Dict[int, Tuple[int, int]],
    input_to_device_map: Dict[int, Tuple[int, int]],
    is_bw: bool,
    logger: Optional[logging.Logger] = None,
):
    """
    Given the workload assignment, input and output relations, group
    the workloads and communication into pipelines with n stages
    """
    # prepare terminal sets for tree packing algo
    n_devices_per_node = len(set(d[1] for d in all_devices))

    def _map_device_to_graph_node_id(d):
        return d[0]

    terminal_sets = []
    costs = []
    output_id_to_lse_id = {}
    for work_id in range(len(workload_costs)):
        output_id = None
        lse_id = None
        for input_id in workload_input_map[work_id]:
            input_meta = input_to_meta_map[input_id]
            if input_meta.type == BlockType.Out:
                output_id = input_id
            elif input_meta.type == BlockType.LSE:
                lse_id = input_id
        assert (output_id is None) == (lse_id is None)
        if output_id is not None:
            output_id_to_lse_id[output_id] = lse_id
    input_required_nodes = defaultdict(set)
    input_required_devices = defaultdict(set)
    skipped_inputs = set()
    tree_id_to_input_id = {}
    for work_id in range(len(workload_costs)):
        assigned_device = workload_to_device_map[work_id]
        for input_id in workload_input_map[work_id]:
            input_meta = input_to_meta_map[input_id]
            if input_meta.type == BlockType.LSE:
                continue
            input_required_nodes[input_id].add(
                _map_device_to_graph_node_id(assigned_device)
            )
            input_required_devices[input_id].add(assigned_device)
    for input_id, devices in input_required_nodes.items():
        input_assigned_device = _map_device_to_graph_node_id(
            input_to_device_map[input_id]
        )
        terminal_set = sorted(
            [d for d in devices if d != input_assigned_device]
        )
        if not terminal_set:
            skipped_inputs.add(input_id)
            continue
        tree_id = len(tree_id_to_input_id)
        tree_id_to_input_id[tree_id] = input_id
        terminal_sets.append((input_assigned_device, terminal_set))
        costs.append(input_sizes[input_id])
    if logger is not None:
        t = time.time()
    trees_per_input = pack_multicast_trees(
        len(set(x[0] for x in all_devices)),
        terminal_sets,
        costs,
        n_stages,
        logger=logger,
    )
    if logger is not None and PIPELINE_LOGGING_ENABLED:
        logger.debug(f"Tree packing total time: {time.time() - t} seconds.")

    # calculate workload that can be executed per stage
    workload_per_stage_per_device = []
    scheduled_workloads = set()
    initial_schedulable_workloads = defaultdict(list)
    for workload_id, d in workload_to_device_map.items():
        if all(
            input_to_device_map[input_id] == d
            for input_id in workload_input_map[workload_id]
        ):
            initial_schedulable_workloads[d].append(workload_id)
            scheduled_workloads.add(workload_id)
    workload_per_stage_per_device.append(initial_schedulable_workloads)

    local_inputs_per_node = defaultdict(set)
    for input_id, d in input_to_device_map.items():
        local_inputs_per_node[d[0]].add(input_id)

    def _get_current_schedulable_workloads_per_device():
        current_schedulable_workloads_per_device = defaultdict(list)
        for workload_id, d in workload_to_device_map.items():
            if workload_id in scheduled_workloads:
                continue
            if all(
                input_id in local_inputs_per_node[d[0]]
                for input_id in workload_input_map[workload_id]
            ):
                current_schedulable_workloads_per_device[d].append(workload_id)
        return current_schedulable_workloads_per_device

    for stage_id in range(n_stages):
        for tree_id, comms_per_stage in enumerate(trees_per_input):
            input_id = tree_id_to_input_id[tree_id]
            src_dst_pairs = comms_per_stage[stage_id]
            for src_node, dst_node in src_dst_pairs:
                local_inputs_per_node[dst_node].add(input_id)
                if input_id in output_id_to_lse_id:
                    local_inputs_per_node[dst_node].add(
                        output_id_to_lse_id[input_id]
                    )
        current_schedulable_workloads_per_device = (
            _get_current_schedulable_workloads_per_device()
        )
        workload_per_stage_per_device.append(
            current_schedulable_workloads_per_device
        )
        # update scheduled workloads
        for d, workloads in current_schedulable_workloads_per_device.items():
            scheduled_workloads.update(workloads)
    # check that all workloads are scheduled
    actual_scheduled_workloads = set()
    for stage_workload_dict in workload_per_stage_per_device:
        for d, workloads in stage_workload_dict.items():
            actual_scheduled_workloads.update(workloads)
    assert actual_scheduled_workloads == set(range(len(workload_costs)))
    # map node level comm schedule to devices intra-node
    input_located_devices = defaultdict(lambda: defaultdict(set))
    for input_id, d in input_to_device_map.items():
        input_located_devices[input_id][d[0]].add(d[1])
    comms_per_stage_per_device = [defaultdict(list)]
    second_phase_comms_per_stage_per_device = [defaultdict(list)]
    local_inputs_per_device = defaultdict(set)
    for input_id, d in input_to_device_map.items():
        local_inputs_per_device[d].add(input_id)
    # now start scheduling
    for stage_id in range(n_stages):
        per_device_comms = defaultdict(list)
        second_phase_per_device_comms = defaultdict(list)
        curr_per_device_send_volume = defaultdict(float)
        curr_per_device_recv_volume = defaultdict(float)
        for tree_id, comms_per_stage in enumerate(trees_per_input):
            input_id = tree_id_to_input_id[tree_id]
            required_devices_per_node = defaultdict(set)
            for d in input_required_devices[input_id]:
                required_devices_per_node[d[0]].add(d[1])
            src_dst_pairs = comms_per_stage[stage_id]
            for src_node, dst_node in src_dst_pairs:
                # find which devices on the src node have the input
                src_node_devices = set()
                for d, inputs in local_inputs_per_device.items():
                    if d[0] == src_node and input_id in inputs:
                        src_node_devices.add(d)
                # choose the device with the minimum send load
                src_device = min(
                    src_node_devices,
                    key=lambda d: curr_per_device_send_volume[d],
                )
                # find which devices on the dst node require the input
                required_devices = required_devices_per_node[dst_node]
                # choose the device with the minimum recv load
                dst_device = min(
                    required_devices,
                    key=lambda d: curr_per_device_recv_volume[d],
                )
                per_device_comms[(dst_node, dst_device)].append(
                    (input_id, src_device)
                )
                local_inputs_per_device[(dst_node, dst_device)].add(input_id)
                if input_id in output_id_to_lse_id:
                    per_device_comms[(dst_node, dst_device)].append(
                        (output_id_to_lse_id[input_id], src_device)
                    )
                    local_inputs_per_device[(dst_node, dst_device)].add(
                        output_id_to_lse_id[input_id]
                    )
                # add second stage comms
                if PIPELINE_DISABLE_TWO_PHASE_COMM:
                    # add to main comm list
                    for dev_id in required_devices:
                        if dev_id != dst_device:
                            per_device_comms[(dst_node, dev_id)].append(
                                (input_id, src_device)
                            )
                            local_inputs_per_device[(dst_node, dev_id)].add(
                                input_id
                            )
                            if input_id in output_id_to_lse_id:
                                per_device_comms[(dst_node, dev_id)].append(
                                    (output_id_to_lse_id[input_id], src_device)
                                )
                                local_inputs_per_device[
                                    (dst_node, dev_id)
                                ].add(output_id_to_lse_id[input_id])
                else:
                    for dev_id in required_devices:
                        if dev_id != dst_device:
                            second_phase_per_device_comms[
                                (dst_node, dev_id)
                            ].append((input_id, (dst_node, dst_device)))
                            local_inputs_per_device[(dst_node, dev_id)].add(
                                input_id
                            )
                            if input_id in output_id_to_lse_id:
                                second_phase_per_device_comms[
                                    (dst_node, dev_id)
                                ].append(
                                    (
                                        output_id_to_lse_id[input_id],
                                        (dst_node, dst_device),
                                    )
                                )
                                local_inputs_per_device[
                                    (dst_node, dev_id)
                                ].add(output_id_to_lse_id[input_id])
        # take care of required intranode comms
        curr_stage_scheduled_workloads = workload_per_stage_per_device[
            stage_id + 1
        ]
        for d, workloads in curr_stage_scheduled_workloads.items():
            required_inputs = set()
            for w in workloads:
                required_inputs.update(workload_input_map[w])
            for input_id in sorted(list(required_inputs)):
                if input_id in local_inputs_per_device[d]:
                    continue
                src_node, src_device = input_to_device_map[input_id]
                assert src_node == d[0]
                per_device_comms[d].append((input_id, (src_node, src_device)))
                local_inputs_per_device[d].add(input_id)
        comms_per_stage_per_device.append(per_device_comms)
        second_phase_comms_per_stage_per_device.append(
            second_phase_per_device_comms
        )

    return (
        workload_per_stage_per_device,
        comms_per_stage_per_device,
        second_phase_comms_per_stage_per_device,
    )
