#!/usr/bin/env bash
source /root/miniconda3/bin/activate dcp

NODE_RANK=$1
NNODES=$2
MASTER_ADDR=$3

if [ $NODE_RANK -eq 0 ]; then
    DATASET="LongAlign"
elif [ $NODE_RANK -eq 1 ]; then
    DATASET="LDC"
else
    echo "Invalid NODE_RANK: $NODE_RANK. Expected 0 or 1."
    exit 1
fi

cd /root/dcp && \
python3 benchmark/plotting/preprocess_dryrun_results.py \
    --exp-dir /root/dcp/dryrun_experiments \
    --out-dir /root/dcp/reproduced_figures/dryrun_preprocess \
    --dataset $DATASET \
    --node-id $NODE_RANK
