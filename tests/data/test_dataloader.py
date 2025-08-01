# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# Note: this test requires torch
# to run this test, exec:
# If running hanging tests or multi-node tests:
# DCP_DEBUG=DEBUG DCP_LOGGING_DEBUG_DIR=./test_debug \
# torchrun --standalone --nnodes=1 --nproc_per_node=4 test_dataloader.py
# Others:
# DCP_DEBUG=DEBUG DCP_LOGGING_DEBUG_DIR=./test_debug \
# torchrun --standalone --nnodes=1 --nproc_per_node=2 test_dataloader.py

import argparse
import os

import torch
import torch.distributed as dist
from torch.utils.data import Dataset

from dcp.core.common import ModelSpec, get_default_execution_context
from dcp.core.instructions import ExecutionPlan
from dcp.data.dataloader import DCPDataLoader, TrainingSpec

torch.manual_seed(42)


def print_all(s):
    for rank in range(dist.get_world_size()):
        if rank == dist.get_rank():
            print(f"RANK {rank}: {s}")
        dist.barrier(device_ids=[torch.cuda.current_device()])


def print_0(s):
    if dist.get_rank() == 0:
        print(s)
    dist.barrier(device_ids=[torch.cuda.current_device()])


def init_env():
    local_rank = os.environ.get("LOCAL_RANK")
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    torch.set_default_device("cpu")
    print_all(f"Rank {dist.get_rank()} set device {device}")


class DummyDataset(Dataset):
    def __init__(self, size):
        self.size = size
        torch.manual_seed(42)
        # pre-generate all data
        self.seqlens = []
        self.data = []
        for _ in range(size):
            seqlen = torch.randint(2**5, 2**13, (1,)).item()
            self.seqlens.append(seqlen)
            result = {
                "text": torch.randint(0, 100, (seqlen,)).cpu().numpy().tolist()
            }
            self.data.append(result)

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        return self.data[index]


def test_joint_data_loader(args):
    world_size = dist.get_world_size()
    n_nodes = args.n_nodes
    # NOTE: This should be for one dp group, instead of the whole world
    exec_context = get_default_execution_context(world_size, n_nodes)

    print(f"exec_context: {exec_context}")

    train_spec = TrainingSpec(
        exec_context,
        ModelSpec(8, 0, 1024, 128, 65536, 128),
        mask_type="causal",
        mask_dtype=3,
        qkv_dtype=1,
        block_size=256,
        head_block_size=32,
        head_dim=128,
        data_context_parallel_size=1,
        tensor_parallel_size=4,
        pipeline_parallel_size=1,
        dcp_size=4,
        n_microbatches=1,
        n_executors=4,
        zero_stage=0,
        device_memory_limit=800000,  # ignore memory limit for this test
    )

    rank = dist.get_rank()
    is_kv_host = rank == 0
    data_loader = DCPDataLoader(
        train_spec,
        DummyDataset(64),
        is_kv_host=is_kv_host,
        node_rank=0,
        node_local_rank=rank,
        dcp_rank=rank,
        pp_rank=0,
        start_poller=rank == 0,
        batch_size=8,
        shuffle=False,
        num_workers=8,
        num_preprocess_workers=8,
        pin_memory=True,
        input_key="text",
    )
    for (
        batch,
        fw_ep,
        bw_ep,
        fw_spec,
        bw_spec,
        training_spec,
        padded_seqlens,
        cumsum_padded_seqlens,
    ) in data_loader:
        assert batch is not None
        assert fw_ep is not None
        assert isinstance(fw_ep, ExecutionPlan)
        assert fw_ep.node_id == 0
        assert fw_ep.local_device_id == rank

    dist.barrier(device_ids=[torch.cuda.current_device()])


if __name__ == "__main__":
    print("Running test_dataloader.py")

    parser = argparse.ArgumentParser(description="Distributed DataLoader Test")
    parser.add_argument(
        "--n_device_per_node",
        type=int,
        default=1,
        help="Number of devices per node",
    )
    parser.add_argument(
        "--n_nodes", type=int, default=1, help="Number of nodes"
    )
    args = parser.parse_args()

    print(f"args: {args}")

    dist.init_process_group(backend="nccl")
    print(f"Rank {dist.get_rank()} initialized process group")

    init_env()

    if args.n_device_per_node > dist.get_world_size():
        args.n_device_per_node = dist.get_world_size()
        if dist.get_rank() == 0:
            print(f"Setting n_device_per_node to {args.n_device_per_node}.")

    print(
        f"Running with {args.n_nodes} nodes, {args.n_device_per_node} devices per node"
    )

    test_joint_data_loader(args)
