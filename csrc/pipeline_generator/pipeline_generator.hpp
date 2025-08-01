#pragma once

#define PYBIND11_DETAILED_ERROR_MESSAGES

#include <pybind11/pybind11.h>
#include <tuple>
#include <vector>
#include <string>
#include <algorithm>
#include <unordered_map>
#include <unordered_set>
#include <stdexcept>
#include <limits>
#include <cassert>

namespace py = pybind11;


namespace dcp::core::pipeline_generator {
using Device = std::tuple<int, int>;
using CommList = std::vector<std::tuple<int, Device>>;
using PipelineStageSpec = std::tuple<std::unordered_map<Device, std::unordered_set<int>>, std::unordered_map<Device, CommList>, std::unordered_map<Device, std::unordered_map<int, int>>>;
}

// define hash function for Device
namespace std {
    template <>
    struct hash<dcp::core::pipeline_generator::Device> {
        std::size_t operator()(const dcp::core::pipeline_generator::Device& d) const {
            return std::hash<int>()(std::get<0>(d)) ^ std::hash<int>()(std::get<1>(d));
        }
    };
} // namespace std

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
);

} // namespace dcp::core::pipeline_generator