#!/bin/bash
# required envs for EFA on aws instances
export FI_PROVIDER="efa"
export FI_EFA_USE_DEVICE_RDMA=1

if [ "$#" -lt 6 ]; then
    echo "Usage: $0 <nnodes> <nproc_per_node> <master_addr> <master_port> <node_rank> <seqlens_config_path>"
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
seqlens_config_paths=( "$@" )

exp_master_port=$((master_port+1))

python -m torch.distributed.launch \
    --nproc_per_node 1 \
    --nnodes ${nnodes} \
    --node_rank ${node_rank} \
    --master_addr ${master_addr} \
    --master_port ${master_port} \
    --use-env \
    benchmark/microbenchmark/run_experiments.py -n ${nnodes} -ng ${nproc_per_node} -m ${master_addr} -p ${exp_master_port} -nr ${node_rank} ${seqlens_config_paths[@]}