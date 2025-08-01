#!/usr/bin/env bash
source /root/miniconda3/bin/activate dcp

NODE_RANK=$1
NNODES=$2
MASTER_ADDR=$3

NGPUS_PER_NODE=8

pkill -f -9 "pretrain_gpt"
pkill -f -9 "envs/dcp/bin/python"
pkill -f -9 "redis-server"
pkill -f -9 "dry_run.py"
sleep 10

# check no previous processes are running
ps -aux | grep -v grep | grep -q pretrain_gpt
if [ $? -eq 0 ]; then
    echo "Previous exp script is still running."
    exit 1
fi

# Check if redis server is still running
ps -aux | grep -v grep | grep redis-server
if [ $? -eq 0 ]; then
    echo "Redis server is still running."
    exit 1
fi

cd /root/dcp && rm -r ./experiments
# check that the directory is removed
if [ -d "./experiments" ]; then
    echo "Directory ./experiments still exists."
else
    echo "Directory ./experiments removed successfully."
fi

# launch experiment script
python3 ./benchmark/mlm/run_experiments.py --num-nodes $NNODES -ip $MASTER_ADDR --n-iters 100 --tp-size 4 --model gpt2-8b --grid-run --dataset THUDM/LongAlign-10k --dataset-text-key messages --dcp-log-schedule --node-rank $NODE_RANK

pkill -f -9 "pretrain_gpt"
pkill -f -9 "envs/dcp/bin/python"
pkill -f -9 "redis-server"
pkill -f -9 "dry_run.py"
sleep 20

python3 ./benchmark/mlm/run_experiments.py --num-nodes $NNODES -ip $MASTER_ADDR --n-iters 100 --tp-size 4 --model gpt2-8b --grid-run --dataset jchenyu/Long-Data-Collections-sample-10000 --dataset-text-key text --dcp-log-schedule --node-rank $NODE_RANK

pkill -f -9 "pretrain_gpt"
pkill -f -9 "envs/dcp/bin/python"
pkill -f -9 "redis-server"
pkill -f -9 "dry_run.py"
