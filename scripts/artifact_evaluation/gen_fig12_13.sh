#!/usr/bin/env bash
source /root/miniconda3/bin/activate dcp

NODE_RANK=$1
NNODES=$2
MASTER_ADDR=$3

cd /root/dcp && \
python3 benchmark/plotting/plot_microbenchmark_results.py \
    --exp-dir /root/dcp/microbenchmarks \
    --out-dir /root/dcp/reproduced_figures/fig12_13
