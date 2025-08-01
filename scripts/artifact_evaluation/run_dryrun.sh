#!/usr/bin/env bash
source /root/miniconda3/bin/activate dcp

export PYTHONPATH=/root/Megatron-LM:$PYTHONPATH

NODE_RANK=$1
NNODES=$2
MASTER_ADDR=$3

NGPUS_PER_NODE=8

if [ $NODE_RANK -eq 0 ]; then
    DATASET="LongAlign"
elif [ $NODE_RANK -eq 1 ]; then
    DATASET="LDC"
else
    echo "Invalid NODE_RANK: $NODE_RANK. Expected 0 or 1."
    exit 1
fi

pkill -f -9 "pretrain_gpt"
pkill -f -9 "envs/dcp/bin/python"
pkill -f -9 "redis-server"
pkill -f -9 "dry_run.py"

cd /root/dcp && rm -r ./dryrun_experiments
# check that the directory is removed
if [ -d "./dryrun_experiments" ]; then
    echo "Directory ./dryrun_experiments still exists."
else
    echo "Directory ./dryrun_experiments removed successfully."
fi

python3 benchmark/mlm/dry_run_grid_search.py --dataset $DATASET