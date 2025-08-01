#!/bin/bash
# required envs for EFA on aws instances
export FI_PROVIDER="efa"
export FI_EFA_USE_DEVICE_RDMA=1

export OMP_NUM_THREADS=1

export DCP_LOG_GRAPH_PARTITION=1
export DCP_DEBUG=DEBUG
export DCP_LOG_INSTRS=1
export DCP_LOG_SCHEDULE=1
# export DCP_LOG_WORKLOAD_SPEC=1
# export NCCL_DEBUG=INFO
# export TORCH_DISTRIBUTED_DEBUG=DETAIL

if [ "$#" -lt 5 ]; then
    echo "Usage: $0 <nnodes> <nproc_per_node> <master_addr> <master_port> <node_rank> args..."
    exit 1
fi

nnodes=$1
shift
nproc_per_node=$1
shift
master_addr=$1
shift
master_port=$1
shift
node_rank=$1
shift

export DCP_KV_HOST=$master_addr
export DCP_KV_PORT=$((master_port + 1))

export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((nproc_per_node - 1)))

python -m torch.distributed.launch \
    --nproc_per_node ${nproc_per_node} \
    --nnodes ${nnodes} \
    --node_rank ${node_rank} \
    --master_addr ${master_addr} \
    --master_port ${master_port} \
    --use-env \
    benchmark/microbenchmark/benchmark_attention.py --profile --profiler-impl torch -nd ${nproc_per_node} "$@"
