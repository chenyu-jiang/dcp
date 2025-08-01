export FI_PROVIDER="efa"
export FI_EFA_USE_DEVICE_RDMA=1
# export NCCL_DEBUG=INFO
# export FI_LOG_LEVEL=WARN

export DCP_DEBUG=DEBUG
# export DCP_LOG_WORKLOAD_SPEC=1
export DCP_LOG_GRAPH_PARTITION=1
export DCP_LOG_SCHEDULE=1

export DCP_LOG_INSTRS=1
export DCP_DEBUG_LOG_EXECUTOR=1


# export DCP_FORCE_SINGLE_STAGE=1

# export CUDA_LAUNCH_BLOCKING=1
# export DCP_DEBUG_SAVE_PARTITION_RESULTS=1
# export DCP_DEBUG_LOAD_PARTITION_RESULTS=1
# export DCP_DEBUG_LOG_REDUCTION_ARGS_ON_ERROR=1
# export DCP_N_STAGES=2
# export DCP_SAVE_EXEC_PLAN_FOR_DEBUG=1
# export DCP_LOAD_EXEC_PLAN_FOR_DEBUG=1

if [  $# -le 3 ]; then
    echo "Usage: $0 <nnodes> <master_addr> <master_port> test_script args"
    exit 1
fi

nnodes=$1
shift
master_addr=$1
shift
master_port=$1
shift

export DCP_KV_HOST=$master_addr
export DCP_KV_PORT=$((master_port + 1))

torchrun \
    --nnodes $nnodes \
    --nproc_per_node 8 \
    --rdzv-backend=c10d \
    --rdzv-endpoint=$master_addr:$master_port \
    "$@"