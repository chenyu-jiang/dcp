#!/usr/bin/env bash
source /root/miniconda3/bin/activate dcp

NODE_RANK=$1
NNODES=$2
MASTER_ADDR=$3

cd /root/dcp && git pull
