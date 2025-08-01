#!/usr/bin/env bash
source /root/miniconda3/bin/activate dcp

NODE_RANK=$1
NNODES=$2
MASTER_ADDR=$3

cd /root/dcp && \
python3 benchmark/plotting/plot_loss_curve.py \
    --exp-dir /root/dcp/experiments \
    --out-dir /root/dcp/reproduced_figures/fig20 \
    --dataset LongAlign \
    --max-seq-len 131072
