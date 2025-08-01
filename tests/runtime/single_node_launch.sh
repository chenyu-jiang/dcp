# if less than one argument supplied, display usage
if [  $# -le 1 ]; then
    echo "Usage: $0 <n_devices> <path_to_test_script> <args>"
    exit 1
fi

export DCP_DEBUG=DEBUG
export DCP_LOG_GRAPH_PARTITION=1
export DCP_LOG_INSTRS=1
export DCP_LOG_SCHEDULE=1
export DCP_DEBUG_LOG_EXECUTOR=1

export OMP_NUM_THREADS=1

# export DCP_DEBUG_SAVE_PARTITION_RESULTS=1
# export DCP_DEBUG_LOAD_PARTITION_RESULTS=1
# export DCP_FORCE_SINGLE_STAGE=1
# export CUDA_LAUNCH_BLOCKING=1

n_devices=$1
shift
script_path=$1
shift

# compute-sanitizer --tool memcheck 
torchrun --standalone --nnodes=1 --nproc-per-node=$n_devices $script_path "$@"