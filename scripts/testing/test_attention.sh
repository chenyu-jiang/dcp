export DCP_DEBUG=DEBUG
# export CUDA_LAUNCH_BLOCKING=1
export DCP_LOG_SCHEDULE=1
export DCP_LOG_INSTRS=1
export DCP_LOG_GRAPH_PARTITION=1
export DCP_DEBUG_LOG_EXECUTOR=1
# export DCP_FORCE_SINGLE_STAGE=1
# export DCP_LOG_WORKLOAD_SPEC=1
export N_DEVICES=4

cd tests/runtime && ./single_node_launch.sh $N_DEVICES test_flash_attention.py -nd $N_DEVICES --head-block-size 2 --attn-mask-type lambda --n-query-groups 4