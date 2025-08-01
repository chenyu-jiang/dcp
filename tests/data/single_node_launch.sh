# if less than one argument supplied, display usage
if [  $# -le 0 ]; then
    echo "Usage: $0 <n_devices> <path_to_test_script> <args>"
    exit 1
fi

n_devices=$1
shift
script_path=$1
shift

# compute-sanitizer --tool memcheck 
torchrun --standalone --nnodes=1 --nproc-per-node=$n_devices $script_path "$@" --n_nodes 1 --n_device_per_node $n_devices