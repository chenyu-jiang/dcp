## Directory Hierachy

```
|--benchmark
|  :contains code for benchmarking and performing experiments.
|   AE will extensively utilize code from this directory.
|
|--csrc
|  :contains the C++ implementation of the computation and communication block
|   scheduling algorithm.
|
|--dcp
|  :contains the DCP python library
|  |--core
|  |  :contains basic and central conponents of DCP, including the planner
|  |  (InstrCompiler in compiler.py), python parts of the scheduler
|  |  (scheduler.py), the 5 types of DCP instructions (Sec 5, instructions.py)
|  |  and the abstract executor (actual executor implementation is located in
|  |  the `runtime` directory).
|  |
|  |--data
|  |  :contains mainly the DCP dataloader
|  |
|  |--graph_partition
|  |  :constructs the hyper-graph and calls partitioning solvers
|  |
|  |--memory
|  |  :implements an caching allocator for pinned memory.
|  |
|  |--ops
|  |  :contains fused block-wise operators implemented with Triton.
|  |
|  |--runtime
|  |  :contains concrete implementation of DCP executor.
|
|--docker
|  :contains the Dockerfile for the experiments
|
|--scripts
|  :various convenience scripts, including the ones used in AE.
|
|--tests
|  :test cases.
```