#include <pybind11/pybind11.h>
#include <tuple>
#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <unordered_map>
#include <unordered_set>
#include <stdexcept>
#include <limits>
#include <cassert>

#include "pipeline_generator.hpp"

namespace py = pybind11;

namespace dcp::core::pipeline_generator {

PipelineStageSpec greedy_selection_per_device(
    const std::unordered_set<Device>& devices,
    const std::unordered_set<int>& scheduled_workloads,
    const std::vector<float>& workload_costs,
    const std::vector<float>& input_sizes,
    const std::vector<std::vector<int>>& workload_input_map,
    const std::vector<std::string>& input_block_type_map,
    const std::unordered_map<Device, std::vector<int>>& device_workload_map,
    const std::unordered_map<Device, std::unordered_map<int, int>>& device_local_inputs_map,
    const std::unordered_map<Device, float>& target_comm_load_per_node,
    bool is_bw,
    int stage_id,
    py::object py_logger
) {
    std::unordered_map<Device, std::unordered_set<int>> unlocked_workloads_per_device;
    std::unordered_map<Device, CommList> performed_comm_per_device;
    std::unordered_map<Device, std::unordered_map<int, int>> updated_device_local_inputs_map = device_local_inputs_map;

    std::unordered_map<Device, float> current_ready_workload_time_per_device;
    std::unordered_map<Device, float> current_cross_node_comm_load_per_device;
    std::unordered_map<Device, float> current_intra_node_comm_load_per_device;

    auto _check_comm_constraints = [&](const std::vector<int>& input_ids, Device dst_device) {
        std::unordered_map<int, Device> internode_required_comms;
        std::unordered_map<int, Device> intranode_required_comms;
        std::unordered_map<Device, float> internode_comm_required_sizes;
        std::unordered_map<Device, float> intranode_comm_required_sizes;
        for (int input_id : input_ids) {
            // first check if the input is local
            if (updated_device_local_inputs_map[dst_device].count(input_id)) {
                continue;
            }
            // requires comm
            // first try to find a local device that has the input
            std::unordered_set<Device> local_devices_having_input;
            for (const auto& [device, local_inputs] : updated_device_local_inputs_map) {
                if (std::get<0>(device) == std::get<0>(dst_device) && local_inputs.count(input_id) && local_inputs.at(input_id) < stage_id) {
                    local_devices_having_input.insert(device);
                }
            }
            if (!local_devices_having_input.empty()) {
                // choose one with minimum intra comm load
                auto min_device = *std::min_element(
                    local_devices_having_input.begin(), local_devices_having_input.end(),
                    [&](const Device& d1, const Device& d2) {
                        return current_intra_node_comm_load_per_device[d1] < current_intra_node_comm_load_per_device[d2];
                    }
                );
                intranode_required_comms[input_id] = min_device;
                if (input_block_type_map[input_id] == "OUT" && is_bw) {
                    // also count dout comm size (which is the same as out)
                    intranode_comm_required_sizes[min_device] += input_sizes[input_id];
                }
                intranode_comm_required_sizes[min_device] += input_sizes[input_id];
                continue;
            }
            // requires internode comm
            std::unordered_set<Device> cross_node_devices_having_input;
            for (const auto& [device, local_inputs] : updated_device_local_inputs_map) {
                if (std::get<0>(device) != std::get<0>(dst_device) && local_inputs.count(input_id) && local_inputs.at(input_id) < stage_id) {
                    cross_node_devices_having_input.insert(device);
                }
            }
            if (!cross_node_devices_having_input.empty()) {
                // choose one with minimum cross comm load
                auto min_device = *std::min_element(
                    cross_node_devices_having_input.begin(), cross_node_devices_having_input.end(),
                    [&](const Device& d1, const Device& d2) {
                        return current_cross_node_comm_load_per_device[d1] < current_cross_node_comm_load_per_device[d2];
                    }
                );
                internode_required_comms[input_id] = min_device;
                if (input_block_type_map[input_id] == "OUT" && is_bw) {
                    // also count dout comm size (which is the same as out)
                    internode_comm_required_sizes[min_device] += input_sizes[input_id];
                }
                internode_comm_required_sizes[min_device] += input_sizes[input_id];
            } else {
                throw std::runtime_error("No device has input.");
            }
        }
        // check constraints (for now we only check internode constraint)
        bool can_perform = true;
        for (const auto& [device, comm_size] : internode_comm_required_sizes) {
            if (current_cross_node_comm_load_per_device[device] + comm_size > target_comm_load_per_node.at(device)) {
                can_perform = false;
                break;
            }
        }
        return std::make_tuple(
            can_perform,
            internode_required_comms, intranode_required_comms,
            internode_comm_required_sizes, intranode_comm_required_sizes
        );
    };

    std::unordered_map<Device, std::unordered_set<int>> pending_workloads_per_device;
    for (const auto& [device, workloads] : device_workload_map) {
        for (int workload_id : workloads) {
            if (scheduled_workloads.count(workload_id) == 0) {
                pending_workloads_per_device[device].insert(workload_id);
            }
        }
    }

    std::unordered_set<Device> devices_to_skip;
    while (true) {
        // choose the device with the minimum ready computation time among non-skipped devices
        Device min_device;
        float min_time = std::numeric_limits<float>::max();
        for (const auto& device: devices) {
            if (devices_to_skip.count(device)) {
                continue;
            }
            float ready_time = current_ready_workload_time_per_device[device];
            if (ready_time < min_time) {
                min_time = ready_time;
                min_device = device;
            }
        }
        // get pending workloads for the device
        // std::vector<int> pending_workloads;
        // for (int workload_id : device_workload_map.at(min_device)) {
        //     if (scheduled_workloads.count(workload_id) == 0 && unlocked_workloads_per_device[min_device].count(workload_id) == 0) {
        //         pending_workloads.push_back(workload_id);
        //     }
        // }
        std::vector<int> pending_workloads(pending_workloads_per_device[min_device].begin(), pending_workloads_per_device[min_device].end());
        // sort by reverse order of workload costs
        std::sort(
            pending_workloads.begin(), pending_workloads.end(),
            [&](int w1, int w2) {
                return workload_costs[w1] > workload_costs[w2];
            }
        );
        if (pending_workloads.empty()) {
            // no more workloads to schedule
            devices_to_skip.insert(min_device);
            if (devices_to_skip.size() == devices.size()) {
                break;
            }
            continue;
        }
        // greedily select the workloads that will not violate communication constraints
        bool progress = false;
        for (int workload_id : pending_workloads) {
            // preprocess inputs to make sure out, lse and dout (in bw) are scheduled together
            std::vector<int> workload_inputs;
            int output_id = -1;
            int lse_id = -1;
            int dout_id = -1;
            for (int input_id : workload_input_map[workload_id]) {
                if (input_block_type_map[input_id] == "OUT") {
                    output_id = input_id;
                } else if (input_block_type_map[input_id] == "LSE") {
                    lse_id = input_id;
                } else if (input_block_type_map[input_id] == "dOUT") {
                    dout_id = input_id;
                } else {
                    workload_inputs.push_back(input_id);
                }
            }
            assert((output_id == -1) == (lse_id == -1));
            if (dout_id != -1) {
                assert(output_id != -1);
            }
            if (output_id != -1) {
                workload_inputs.push_back(output_id);
            }
            // check comm constraints
            auto [
                can_perform, internode_required_comms, intranode_required_comms,
                internode_comm_required_sizes, intranode_comm_required_sizes
            ] = _check_comm_constraints(workload_inputs, min_device);
            if (can_perform) {
                // schedule the workload
                unlocked_workloads_per_device[min_device].insert(workload_id);
                // update comm loads
                for (const auto& [input_id, src_device] : internode_required_comms) {
                    performed_comm_per_device[min_device].push_back(std::make_tuple(input_id, src_device));
                    if (input_id == output_id) {
                        performed_comm_per_device[min_device].push_back(std::make_tuple(lse_id, src_device));
                        if (dout_id != -1) {
                            performed_comm_per_device[min_device].push_back(std::make_tuple(dout_id, src_device));
                        }
                    }
                }
                for (const auto& [input_id, src_device] : intranode_required_comms) {
                    performed_comm_per_device[min_device].push_back(std::make_tuple(input_id, src_device));
                    if (input_id == output_id) {
                        performed_comm_per_device[min_device].push_back(std::make_tuple(lse_id, src_device));
                        if (dout_id != -1) {
                            performed_comm_per_device[min_device].push_back(std::make_tuple(dout_id, src_device));
                        }
                    }
                }
                for (const auto& [device, comm_size] : internode_comm_required_sizes) {
                    current_cross_node_comm_load_per_device[device] += comm_size;
                }
                for (const auto& [device, comm_size] : intranode_comm_required_sizes) {
                    current_intra_node_comm_load_per_device[device] += comm_size;
                }
                for (auto input_id : workload_input_map[workload_id]) {
                    if (!updated_device_local_inputs_map[min_device].count(input_id)) {
                        updated_device_local_inputs_map[min_device][input_id] = stage_id;
                    }
                }
                // update ready time
                current_ready_workload_time_per_device[min_device] += workload_costs[workload_id];
                pending_workloads_per_device[min_device].erase(workload_id);
                progress = true;
                break;
            }
        }
        if (!progress) {
            break;
        }
    }
    // next, squeeze in addtional communication if possible
    for (auto device: devices) {
        std::vector<int> remaining_workloads(pending_workloads_per_device[device].begin(), pending_workloads_per_device[device].end());
        std::sort(
            remaining_workloads.begin(), remaining_workloads.end(),
            [&](int w1, int w2) {
                return workload_costs[w1] > workload_costs[w2];
            }
        );
        for (int workload: remaining_workloads) {
            int output_id = -1;
            int lse_id = -1;
            int dout_id = -1;
            for (int input_id : workload_input_map[workload]) {
                if (input_block_type_map[input_id] == "OUT") {
                    output_id = input_id;
                } else if (input_block_type_map[input_id] == "LSE") {
                    lse_id = input_id;
                } else if (input_block_type_map[input_id] == "dOUT") {
                    dout_id = input_id;
                }
            }
            assert((output_id == -1) == (lse_id == -1));
            if (dout_id != -1) {
                assert(output_id != -1);
            }
            for (int input_id: workload_input_map[workload]) {
                if (input_id == lse_id || input_id == dout_id) {
                    continue;
                }
                if (updated_device_local_inputs_map[device].count(input_id)) {
                    continue;
                }
                // check comm constraints
                auto [
                    can_perform, internode_required_comms, intranode_required_comms,
                    internode_comm_required_sizes, intranode_comm_required_sizes
                ] = _check_comm_constraints({input_id}, device);
                if (can_perform) {
                    // schedule the comm
                    // update comm loads
                    for (const auto& [input_id, src_device] : internode_required_comms) {
                        performed_comm_per_device[device].push_back(std::make_tuple(input_id, src_device));
                        if (input_id == output_id) {
                            performed_comm_per_device[device].push_back(std::make_tuple(lse_id, src_device));
                            if (dout_id != -1) {
                                performed_comm_per_device[device].push_back(std::make_tuple(dout_id, src_device));
                            }
                        }
                    }
                    for (const auto& [input_id, src_device] : intranode_required_comms) {
                        performed_comm_per_device[device].push_back(std::make_tuple(input_id, src_device));
                        if (input_id == output_id) {
                            performed_comm_per_device[device].push_back(std::make_tuple(lse_id, src_device));
                            if (dout_id != -1) {
                                performed_comm_per_device[device].push_back(std::make_tuple(dout_id, src_device));
                            }
                        }
                    }
                    for (const auto& [src_device, comm_size] : internode_comm_required_sizes) {
                        current_cross_node_comm_load_per_device[src_device] += comm_size;
                    }
                    for (const auto& [src_device, comm_size] : intranode_comm_required_sizes) {
                        current_intra_node_comm_load_per_device[src_device] += comm_size;
                    }
                    updated_device_local_inputs_map[device][input_id] = stage_id;
                    if (input_id == output_id) {
                        updated_device_local_inputs_map[device][lse_id] = stage_id;
                        if (dout_id != -1) {
                            updated_device_local_inputs_map[device][dout_id] = stage_id;
                        }
                    }
                }
            }
        }
    }
    if (!py_logger.is_none()) {
        for (auto d: devices) {
            py_logger.attr("debug")(
                "Current communication load for device (" + std::to_string(std::get<0>(d)) + ", " + std::to_string(std::get<1>(d)) + "):" +
                " intra_node=" + std::to_string(current_intra_node_comm_load_per_device[d]) + " (target: " + std::to_string(0) + ")," +
                " inter_node=" + std::to_string(current_cross_node_comm_load_per_device[d]) + " (target: " + std::to_string(target_comm_load_per_node.at(d)) + ")"
            );
        }
    }
    return std::make_tuple(unlocked_workloads_per_device, performed_comm_per_device, updated_device_local_inputs_map);
}
} // namespace dcp::core::pipeline_generator