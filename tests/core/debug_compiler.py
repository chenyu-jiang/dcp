import os

os.environ["DCP_DEBUG"] = "DEBUG"
os.environ["DCP_LOG_GRAPH_PARTITION"] = "1"
# os.environ["DCP_DEBUG_VISUALIZE_WORKLOAD_SPEC"] = "1"
# os.environ["DCP_DEBUG_VISUALIZE_WORKLOAD_SPEC_DIR"] = "./workload_spec"
os.environ["DCP_LOG_SCHEDULE"] = "1"
os.environ["DCP_LOG_PIPELINE"] = "1"
# os.environ["DCP_DEBUG_OVERRIDE_PARTITION_WITH_HEURISTIC"] = "1"
# os.environ["DCP_DISABLE_TWO_PHASE_COMM"] = "1"
# os.environ["DCP_LOG_INSTRS"] = "1"

from dcp.core.compiler import InstrCompiler, CompilerConfig
from dcp.core.common import ExecutionContext
from dcp.core.cost_model import (
    AttnRooflineCostModel,
    CommunicationCostModel,
)
from dcp.data.dataloader import _generate_workload
from dcp.runtime.flash_attention.utils import (
    causal_range_mask_fn,
    lambda_mask_fn,
)
from dcp.graph_partition.graph import HGPSolver

import torch
import torch.nn.functional as F

raw_seqlens = [45613, 9419]
padded_seqlens = [46080, 10240]

mask_dtype = torch.int32

context = ExecutionContext(
    2,
    4,
    CommunicationCostModel(),
    AttnRooflineCostModel(),
)
config = CompilerConfig(
    mem_imbalance_epsilon=0.2,
    comp_imbalance_epsilon=0.2,
    inter_node_comp_imbalance_factor=5,
    solver=HGPSolver.KAHYPAR,
)
compiler = InstrCompiler(
    context,
    config,
    iteration_idx=0,
)

workload = _generate_workload(
    q_shape=(sum(raw_seqlens), 8, 128),
    kv_shape=(sum(raw_seqlens), 2, 2, 128),
    qkv_dtype=torch.bfloat16,
    raw_seqlens=raw_seqlens,
    padded_seqlens=padded_seqlens,
    block_size=1024,
    head_block_size=1,
    n_total_devices=8,
    n_devices_per_node=2,
    comp_cost_model=AttnRooflineCostModel(),
    attn_mask=causal_range_mask_fn(raw_seqlens, padded_seqlens, mask_dtype),
)

compiler.compile(workload, generate_backward=False)
