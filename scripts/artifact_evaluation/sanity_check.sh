#!/usr/bin/env bash
source /root/miniconda3/bin/activate dcp
cd /root/dcp

./benchmark/microbenchmark/run_distributed_benchmark.sh 1 8 $(hostname -i) 9876 0 -b ring
./benchmark/microbenchmark/run_distributed_benchmark.sh 1 8 $(hostname -i) 9876 0 -b zigzag
./benchmark/microbenchmark/run_distributed_benchmark.sh 1 8 $(hostname -i) 9876 0 -b te
./benchmark/microbenchmark/run_distributed_benchmark.sh 1 8 $(hostname -i) 9876 0 -b lt
./benchmark/microbenchmark/run_distributed_benchmark.sh 1 8 $(hostname -i) 9876 0 -b dcp