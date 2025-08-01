#!/usr/bin/env bash
source /root/miniconda3/bin/activate dcp

NODE_RANK=$1
NNODES=$2
MASTER_ADDR=$3

NGPUS_PER_NODE=8

pkill -f "benchmark_attention"
pkill redis-server
sleep 10

# check no previous processes are running
ps -aux | grep -v grep | grep -q benchmark_attention
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

cd /root/dcp && rm -r ./microbenchmarks
# check that the directory is removed
if [ -d "./microbenchmarks" ]; then
    echo "Directory ./microbenchmarks still exists."
else
    echo "Directory ./microbenchmarks removed successfully."
fi

# generate samples for microbenchmarks
# this generates the sample json files under
# /root/dcp/benchmark/preprocessing/<Dataset>/ directories
pushd ./benchmark/preprocessing/LongAlign && \
    python3 get_longalign_seqlens.py && \
    python3 get_longalign_variations.py && \
    python3 get_longalign_batches.py && \
    popd
pushd ./benchmark/preprocessing/LongDataCollection && \
    python3 get_ldc_seqlens.py && \
    python3 get_ldc_variations.py && \
    python3 get_ldc_batches.py && \
    popd

# launch experiment script
./benchmark/microbenchmark/run_distributed_exp.sh $NNODES $NGPUS_PER_NODE $MASTER_ADDR 9876 $NODE_RANK /root/dcp/benchmark/preprocessing/LongDataCollection/*.json -ns 50

pkill -f -9 "benchmark_attention"
pkill redis-server
sleep 20

./benchmark/microbenchmark/run_distributed_exp.sh $NNODES $NGPUS_PER_NODE $MASTER_ADDR 9876 $NODE_RANK /root/dcp/benchmark/preprocessing/LongAlign/*.json -ns 50

pkill -f -9 "benchmark_attention"
pkill redis-server
