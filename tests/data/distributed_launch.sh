export FI_PROVIDER="efa"
export FI_EFA_USE_DEVICE_RDMA=1
export RDMAV_FORK_SAFE=1
export NCCL_PXN_DISABLE=0

if [  $# -le 4 ]; then
    echo "Usage: $0 <nnodes> <master_addr> <master_port> <node_rank> test_script args"
    exit 1
fi

nnodes=$1
shift
master_addr=$1
shift
master_port=$1
shift
node_rank=$1
shift

python -m torch.distributed.launch \
    --nproc_per_node 8 \
    --nnodes $nnodes \
    --node_rank $node_rank \
    --master_addr $master_addr \
    --master_port $master_port \
    --use-env \
    "$@" --n_nodes $nnodes --n_device_per_node 8