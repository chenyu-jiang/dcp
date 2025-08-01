#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include "./pipeline_generator/pipeline_generator.hpp"

namespace py = pybind11;

// expose the functions to Python
PYBIND11_MODULE(dcp_cpp, m) {
  m.def("greedy_selection_per_device", &dcp::core::pipeline_generator::greedy_selection_per_device);
}