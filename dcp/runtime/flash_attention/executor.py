import os
import pickle
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional, Set, Tuple

from dcp.utils.common import trepr
import dcp_flash_attn
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from dcp_flash_attn.flash_attn_interface import (
    _flash_attn_varlen_backward,
    _flash_attn_varlen_forward,
)
from packaging.version import Version as PkgVersion

from dcp.core.block_table import BlockType, WorkloadSpec
from dcp.core.common import ExecutionContext
from dcp.core.executor import DCPExecutor
from dcp.core.instructions import *  # noqa: F403
from dcp.core.serialization import int_to_torch_dtype, torch_dtype_to_int
from dcp.ops.fused_ops import (
    flash_attn_update_out_lse,
    fused_copy_bf16_varlen,
    fused_copy_fp16_varlen,
    fused_copy_fp32_varlen,
    fused_copy_bf16_varlen_dkv,
    fused_copy_fp16_varlen_dkv,
    fused_copy_fp32_varlen_dkv,
    fused_copy_fp32_varlen_lse,
    fused_blockwise_sum,
)
from dcp.utils.logger import read_env_bool

flash_attn_2_5_8_plus = PkgVersion(dcp_flash_attn.__version__) > PkgVersion(
    "2.5.8"
)

if not flash_attn_2_5_8_plus:
    try:
        from ring_flash_attn.triton_utils import unflatten_varlen_lse
    except:
        from ring_flash_attn.utils import unflatten_varlen_lse


DEBUG_LOG_REDUCTION_ARGS_ON_ERROR = read_env_bool(
    "DCP_DEBUG_LOG_REDUCTION_ARGS_ON_ERROR", default=False
)

IS_L40 = False  # "L40" in torch.cuda.get_device_name(0)


def get_byte_tensor(b: bytes):
    return torch.tensor(
        list(b), dtype=torch.uint8, device=torch.get_default_device()
    )


def read_byte_from_tensor(t: torch.Tensor):
    return bytes(t.cpu().tolist())


def _wrapped_flash_attn_varlen_forward_for_te(
    q,
    k,
    v,
    cu_seqlens_q,
    cu_seqlens_k,
    total_q,
    max_seqlen_q,
    max_seqlen_k,
    attn_mask,
):
    # hard code some parameters here
    dropout_p = 0.0
    softmax_scale = None
    causal = False
    window_size = (-1, -1)
    softcap = 0.0
    alibi_slopes = None
    deterministic = False
    return_softmax = True

    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** (-0.5)
    head_size_og = q.size(2)
    if head_size_og % 8 != 0:
        q = torch.nn.functional.pad(q, [0, 8 - head_size_og % 8])
        k = torch.nn.functional.pad(k, [0, 8 - head_size_og % 8])
        v = torch.nn.functional.pad(v, [0, 8 - head_size_og % 8])
    out_padded, softmax_lse, S_dmask, rng_state = _flash_attn_varlen_forward(
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        total_q,
        max_seqlen_q,
        max_seqlen_k,
        dropout_p,
        softmax_scale,
        causal=causal,
        window_size_left=window_size[0],
        window_size_right=window_size[1],
        softcap=softcap,
        alibi_slopes=alibi_slopes,
        return_softmax=return_softmax and dropout_p > 0,
        attn_range=attn_mask,
    )
    out = out_padded[..., :head_size_og]
    # return out if not return_softmax else (out, softmax_lse, S_dmask, rng_state)
    return out, softmax_lse, rng_state


def _wrapped_flash_attn_varlen_forward(
    q,
    kv,
    cu_seqlens_q,
    cu_seqlens_k,
    total_q,
    max_seqlen_q,
    max_seqlen_k,
    # dropout_p,
    # softmax_scale,
    # causal,
    # window_size,
    # softcap,
    # alibi_slopes,
    # deterministic,
    # return_softmax,
    attn_mask,
    q_block_table,
    kv_block_table,
    out_block_table,
    out_,
    lse_,
):
    # hard code some parameters here
    dropout_p = 0.0
    softmax_scale = None
    causal = False
    window_size = (-1, -1)
    softcap = 0.0
    alibi_slopes = None
    deterministic = False
    return_softmax = True

    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** (-0.5)
    if kv_block_table is None:
        k, v = kv[:, 0].detach(), kv[:, 1].detach()
    else:
        k, v = kv[:, :, 0].detach(), kv[:, :, 1].detach()
    head_size_og = q.size(2) if q_block_table is None else q.size(3)
    if head_size_og % 8 != 0:
        q = torch.nn.functional.pad(q, [0, 8 - head_size_og % 8])
        k = torch.nn.functional.pad(k, [0, 8 - head_size_og % 8])
        v = torch.nn.functional.pad(v, [0, 8 - head_size_og % 8])
    out_padded, softmax_lse, S_dmask, rng_state = _flash_attn_varlen_forward(
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        total_q,
        max_seqlen_q,
        max_seqlen_k,
        dropout_p,
        softmax_scale,
        causal=causal,
        window_size_left=window_size[0],
        window_size_right=window_size[1],
        softcap=softcap,
        alibi_slopes=alibi_slopes,
        return_softmax=return_softmax and dropout_p > 0,
        attn_range=attn_mask,
        q_block_table=q_block_table,
        kv_block_table=kv_block_table,
        out_block_table=out_block_table,
        out_=out_,
        lse_=lse_,
    )
    out = out_padded[..., :head_size_og]
    # return out if not return_softmax else (out, softmax_lse, S_dmask, rng_state)
    return out, softmax_lse, rng_state


def _wrapped_flash_attn_varlen_backward_for_te(
    dout,
    q,
    k,
    v,
    out,
    lse,
    cu_seqlens_q,
    cu_seqlens_k,
    total_q,
    total_k,
    max_seqlen_q,
    max_seqlen_k,
    attn_mask,
    dq,
    dk,
    dv,
):
    # hard code some parameters here
    dropout_p = 0.0
    softmax_scale = None
    causal = False
    window_size = (-1, -1)
    softcap = 0.0
    alibi_slopes = None
    deterministic = False

    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** (-0.5)
    head_size_og = dout.size(2)
    dout_padded = dout
    if head_size_og % 8 != 0:
        dout_padded = torch.nn.functional.pad(dout, [0, 8 - head_size_og % 8])
    _flash_attn_varlen_backward(
        dout_padded,
        q,
        k,
        v,
        out,
        lse,
        dq,
        dk,
        dv,
        cu_seqlens_q,
        cu_seqlens_k,
        total_q,
        total_k,
        max_seqlen_q,
        max_seqlen_k,
        dropout_p,
        softmax_scale,
        causal,
        window_size[0],
        window_size[1],
        softcap,
        alibi_slopes,
        deterministic,
        attn_range=attn_mask,
        # rng_state=rng_state,
    )
    dq = dq[..., : dout.shape[-1]]  # We could have padded the head dimension
    dk = dk[..., : dout.shape[-1]]
    dv = dv[..., : dout.shape[-1]]
    return dq, dk, dv


def _wrapped_flash_attn_varlen_backward(
    dout,
    q,
    kv,
    out,
    lse,
    cu_seqlens_q,
    cu_seqlens_k,
    total_q,
    total_k,
    max_seqlen_q,
    max_seqlen_k,
    # dropout_p,
    # softmax_scale,
    # causal,
    # window_size,
    # softcap,
    # alibi_slopes,
    # deterministic,
    attn_mask,
    q_block_table,
    kv_block_table,
    out_block_table,
    dq_block_table,
    dkv_block_table,
    # rng_state,
    dq,
    dkv,
):
    # hard code some parameters here
    dropout_p = 0.0
    softmax_scale = None
    causal = False
    window_size = (-1, -1)
    softcap = 0.0
    alibi_slopes = None
    deterministic = False

    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** (-0.5)
    k, v = kv[:, :, 0].detach(), kv[:, :, 1].detach()
    dk = dkv[:, :, 0]
    dv = dkv[:, :, 1]
    head_size_og = dout.size(2) if out_block_table is None else dout.size(3)
    dout_padded = dout
    if head_size_og % 8 != 0:
        dout_padded = torch.nn.functional.pad(dout, [0, 8 - head_size_og % 8])
    _flash_attn_varlen_backward(
        dout_padded,
        q,
        k,
        v,
        out,
        lse,
        dq,
        dk,
        dv,
        cu_seqlens_q,
        cu_seqlens_k,
        total_q,
        total_k,
        max_seqlen_q,
        max_seqlen_k,
        dropout_p,
        softmax_scale,
        causal,
        window_size[0],
        window_size[1],
        softcap,
        alibi_slopes,
        deterministic,
        attn_range=attn_mask,
        q_block_table=q_block_table,
        kv_block_table=kv_block_table,
        out_block_table=out_block_table,
        dq_block_table=dq_block_table,
        dkv_block_table=dkv_block_table,
        # rng_state=rng_state,
    )
    dq = dq[..., : dout.shape[-1]]  # We could have padded the head dimension
    dkv = dkv[..., : dout.shape[-1]]
    return dq, dkv


class AttentionExecutor(DCPExecutor):
    def __init__(
        self,
        fw_exec_plan: Optional[ExecutionPlan] = None,
        fw_workload: Optional[WorkloadSpec] = None,
        bw_exec_plan: Optional[ExecutionPlan] = None,
        bw_workload: Optional[WorkloadSpec] = None,
        node_id: int = None,
        local_device_id: int = None,
        n_devices_per_node: int = None,
        n_nodes: int = None,
        n_heads_per_block: int = 1,
        head_dim: int = None,
        synchronous: bool = False,
        process_group: Optional[dist.ProcessGroup] = None,
        iteration_idx: Optional[int] = None,
        tp_rank: Optional[int] = None,
        use_cudagraph: bool = False,
    ):
        super().__init__(
            node_id=node_id,
            local_device_id=local_device_id,
            n_devices_per_node=n_devices_per_node,
            n_nodes=n_nodes,
            synchronous=synchronous,
            iteration_idx=iteration_idx,
            tp_rank=tp_rank,
        )
        self.buffers: Dict[str, torch.Tensor]
        if process_group is None:
            process_group = dist.group.WORLD
        self.process_group = process_group
        self.pending_comm_ops: Dict[str, Tuple] = {}
        self.n_heads_per_block = n_heads_per_block
        self.head_dim = head_dim
        self.main_stream = torch.cuda.Stream()
        self.streams = {}
        self._prep_handlers = {}
        self._args_cache = {}
        self._gqa_dkv_args = {}
        # self.rng_state_cache = {} # cache rng state for each work_id
        self.register_synchronization_handler(torch.cuda.synchronize)

        self.fw_exec_plan = fw_exec_plan
        self.bw_exec_plan = bw_exec_plan
        self.fw_workload = fw_workload
        self.bw_workload = bw_workload

        self.orig_block_offset_to_offset = None

        self.qkv_shape = None
        self.qkv_dtype = None
        self.mode = "Forward"
        self.use_cudagraph = use_cudagraph
        self.cuda_graph = None

    def _create_buffer(self, buffer_shape: Tuple[int, ...], buffer_dtype: int):
        return torch.zeros(
            *buffer_shape,
            dtype=int_to_torch_dtype(buffer_dtype),
            device="cuda",
        )

    def capture_cudagraph(self):
        if not self.use_cudagraph:
            return
        # capture cudagraph must be called after prepare
        # to avoid triton compilation, call it after the first run
        if self.logger:
            self.logger.debug("Capturing cudagraph...")
        if self.mode == "Forward":
            exec_plan = self.fw_exec_plan
            is_forward = True
        else:
            exec_plan = self.bw_exec_plan
            is_forward = False
        self.cuda_graph = None
        g = torch.cuda.CUDAGraph()
        # capture
        with torch.cuda.graph(g):
            self.execute(exec_plan, is_forward=is_forward)
        if self.logger:
            self.logger.debug("Cudagraph captured.")
        torch.cuda.synchronize()
        self.cuda_graph = g

    def log_memory_usage(self):
        total_memory = 0
        for buffer_type, buffer in self.buffers.items():
            mem_size = buffer.element_size() * buffer.numel() / 1e6
            total_memory += mem_size
            if self.logger:
                self.logger.debug(f"Buffer {buffer_type}: {mem_size:.2f} MB")
        if self.logger:
            self.logger.debug(f"Total memory: {total_memory:.2f} MB")

    def register_prep_handler(
        self, instruction_type: Type[BlockInstrBase] | Any, handler
    ):
        if not issubclass(instruction_type, BlockInstrBase):
            raise TypeError(
                f"Instruction type must be a subclass of BlockInstruction, "
                f"got {instruction_type.__name__}"
            )
        if instruction_type in self._prep_handlers:
            raise ValueError(
                f"Prep handler for {instruction_type.__name__} "
                "already registered."
            )
        self._prep_handlers[instruction_type] = handler

    def set_local_out_lse(self, out, lse):
        self.buffers[BufferType.LOCAL_OUT].copy_(out)
        self.buffers[BufferType.LOCAL_LSE].copy_(lse)

    def prepare(self, forward=True):
        torch.cuda.nvtx.range_push("AttentionExecutor prepare")
        self._args_cache = {}
        self._gqa_dkv_args = {}
        if forward:
            self.execution_plan = self.fw_exec_plan
            self.init_buffers(self.fw_exec_plan)
        else:
            self.execution_plan = self.bw_exec_plan
            self.init_buffers(self.bw_exec_plan)
        for instr_index, instr in enumerate(self.execution_plan.instructions):
            self.instr_index = instr_index
            if type(instr) in self._prep_handlers:
                self._prep_handlers[type(instr)](self, instr)
        torch.cuda.nvtx.range_pop()

    def switch_to_backward(self):
        if self.logger:
            self.logger.debug("Switching to backward")
        self.mode = "Backward"
        self.cuda_graph = None
        self.deallocate_buffers()
        self.prepare(forward=False)
        self.log_memory_usage()

    def deallocate_all_buffers(self):
        self.buffers = {}
        self.streams = {}
        self._args_cache = {}
        self._gqa_dkv_args = {}

    def execute(self, execution_plan, is_forward=True):
        torch.cuda.nvtx.range_push(
            f"AttentionExecutor DCP({self.node_id}, {self.local_device_id}),TP({self.tp_rank}) Iter({self.iteration_idx})"
        )
        # move everything to a non-default stream
        prev_stream = torch.cuda.current_stream()
        self.main_stream.wait_stream(prev_stream)
        if self.use_cudagraph and self.cuda_graph is None:
            # all created streams should wait on the default stream
            # so they are captured in the cudagraph
            for stream in self.streams.values():
                stream.wait_stream(prev_stream)
        with torch.cuda.stream(self.main_stream):
            if self.cuda_graph is not None:
                self.cuda_graph.replay()
            else:
                super().execute(execution_plan, is_forward)
        # make default stream wait on the main stream
        prev_stream.wait_stream(self.main_stream)
        torch.cuda.nvtx.range_pop()
        if is_forward:
            return (
                self.buffers[BufferType.LOCAL_OUT],
                self.buffers[BufferType.LOCAL_LSE],
            )
        else:
            return (
                self.buffers[BufferType.LOCAL_dQ],
                self.buffers[BufferType.LOCAL_dKV],
            )

    def forward(self, q, kv):
        self.init_forward_input(q, kv)
        self.execute(self.fw_exec_plan, is_forward=True)
        return (
            self.buffers[BufferType.LOCAL_OUT].detach().clone(),
            self.buffers[BufferType.LOCAL_LSE].detach().clone(),
        )

    def backward(self, q, kv, out, dout, lse):
        self.set_local_out_lse(out, lse)
        self.init_forward_input(q, kv)
        self.init_backward_input(dout)
        self.execute(self.bw_exec_plan, is_forward=False)
        return (
            self.buffers[BufferType.LOCAL_dQ].detach().clone(),
            self.buffers[BufferType.LOCAL_dKV].detach().clone(),
        )


class DCPAttention(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        executor: AttentionExecutor,
        capture_fwd_graph: bool,
        capture_bwd_graph: bool,
        q: torch.Tensor,
        kv: torch.Tensor,
    ):
        ctx.executor = executor
        ctx.capture_bwd_graph = capture_bwd_graph
        if capture_fwd_graph:
            executor.capture_cudagraph()
        out, lse = executor.forward(q, kv)
        ctx.save_for_backward(q, kv, out, lse)
        return out

    @staticmethod
    def backward(ctx: torch.autograd.function.FunctionCtx, dout: torch.Tensor):
        q, kv, out, lse = ctx.saved_tensors
        executor: AttentionExecutor = ctx.executor
        # if torch.isnan(dout).any() or torch.isinf(dout).any():
        #     executor.logger.error(
        #         "DCPAttention dout contains NaNs/Infs."
        #     )
        # else:
        #     executor.logger.debug(
        #         "DCPAttention dout norm: {:.4f}".format(dout.norm())
        #     )
        if executor.mode == "Forward":
            executor.switch_to_backward()
        if ctx.capture_bwd_graph:
            executor.capture_cudagraph()
        dq, dkv = executor.backward(q, kv, out, dout, lse)
        # if torch.isnan(dq).any() or torch.isinf(dq).any():
        #     executor.logger.error(
        #         "DCPAttention dq contains NaNs/Infs."
        #     )
        # else:
        #     executor.logger.debug(
        #         "DCPAttention dq norm: {:.4f}".format(dq.norm())
        #     )
        # if torch.isnan(dkv).any() or torch.isinf(dkv).any():
        #     executor.logger.error(
        #         "DCPAttention dkv contains NaNs/Infs."
        #     )
        # else:
        #     executor.logger.debug(
        #         "DCPAttention dkv norm: {:.4f}".format(dkv.norm())
        #     )
        return None, None, None, dq, dkv


_LSE_BUFFER_TYPES = {
    BufferType.BUFFER_LSE,
    BufferType.LOCAL_LSE,
}

_LOCAL_UNPADDED_BUFFER_TYPES = {
    BufferType.LOCAL_Q,
    BufferType.LOCAL_KV,
    BufferType.LOCAL_OUT,
    BufferType.LOCAL_dQ,
    BufferType.LOCAL_dKV,
    BufferType.LOCAL_dOUT,
    BufferType.LOCAL_LSE,
}


def _prepare_memcpy(exec: AttentionExecutor, instr: MemcpyInstr):
    torch.cuda.nvtx.range_push("Prepare MemcpyInstr")
    instr_index = exec.instr_index
    # separate each dtype
    src_ptrs = defaultdict(list)
    dst_ptrs = defaultdict(list)
    src_offsets = defaultdict(list)
    dst_offsets = defaultdict(list)
    src_strides = defaultdict(list)
    dst_strides = defaultdict(list)
    n_elems = defaultdict(list)
    # separate lse and non-lse buffer types
    lse_src_ptrs = defaultdict(list)
    lse_src_token_offsets = defaultdict(list)
    lse_src_head_strides = defaultdict(list)
    lse_dst_ptrs = defaultdict(list)
    lse_dst_token_offsets = defaultdict(list)
    lse_dst_head_strides = defaultdict(list)
    lse_n_tokens = defaultdict(list)
    head_block_size = None

    local_cu_seqlens = exec.execution_plan.local_cu_seqlens
    for src_block, dst_block in instr.src_dst_pairs:
        src_dtype = exec.buffers[src_block.buffer_type].dtype
        dst_dtype = exec.buffers[dst_block.buffer_type].dtype
        src_block_size = exec.buffer_block_strides[src_block.buffer_type]
        dst_block_size = exec.buffer_block_strides[dst_block.buffer_type]
        src_n_tokens = src_block.n_tokens
        dst_n_tokens = dst_block.n_tokens
        # here src is the unpadded buffer, so we use local_cu_seqlens offset
        src_offset = local_cu_seqlens[src_block.index]
        if src_block.buffer_type in _LSE_BUFFER_TYPES:
            lse_src_ptrs[(src_dtype, src_block_size)].append(
                exec.buffers[src_block.buffer_type].data_ptr()
            )
            lse_src_token_offsets[(src_dtype, src_block_size)].append(
                src_offset
            )
            lse_src_head_strides[(src_dtype, src_block_size)].append(
                exec.buffers[src_block.buffer_type].shape[1]
            )
            lse_dst_ptrs[(dst_dtype, dst_block_size)].append(
                exec.buffers[dst_block.buffer_type].data_ptr()
            )
            lse_dst_token_offsets[(dst_dtype, dst_block_size)].append(
                dst_block.index * exec.buffers[dst_block.buffer_type].shape[2]
            )
            lse_dst_head_strides[(dst_dtype, dst_block_size)].append(
                exec.buffers[dst_block.buffer_type].shape[1]
                * exec.buffers[dst_block.buffer_type].shape[2]
            )
            lse_n_tokens[(src_dtype, src_block_size)].append(src_n_tokens)
            if head_block_size is None:
                head_block_size = exec.buffers[src_block.buffer_type].shape[0]
        else:
            assert (
                src_dtype == dst_dtype
            ), f"{src_dtype} != {dst_dtype} in MemcpyInstr"
            assert src_dtype in [
                torch.float32,
                torch.float16,
                torch.bfloat16,
            ], f"Unsupported dtype {src_dtype}"
            assert (
                src_block_size == dst_block_size
            ), f"{src_block_size} != {dst_block_size} in MemcpyInstr"
            assert (
                src_n_tokens == dst_n_tokens
            ), f"{src_n_tokens} != {dst_n_tokens} in MemcpyInstr"
            if src_block.buffer_type in _LSE_BUFFER_TYPES:
                src_buffer_tensor = exec.buffers[src_block.buffer_type]
                src_ptrs[(src_dtype, src_block_size)].append(
                    # exec.buffers[src_block.buffer_type][:, src_offset:].data_ptr()
                    src_buffer_tensor.data_ptr()
                    + src_offset
                    * src_buffer_tensor.stride(1)
                    * exec.buffers[src_block.buffer_type].element_size()
                )
                dst_ptrs[(src_dtype, src_block_size)].append(
                    exec.buffers[dst_block.buffer_type].data_ptr()
                )
                n_elems[(src_dtype, src_block_size)].append(src_n_tokens)
            else:
                src_buffer_tensor = exec.buffers[src_block.buffer_type]
                src_ptrs[(src_dtype, src_block_size)].append(
                    # exec.buffers[src_block.buffer_type][src_offset:].data_ptr()
                    src_buffer_tensor.data_ptr()
                    + src_offset
                    * src_buffer_tensor.stride(0)
                    * exec.buffers[src_block.buffer_type].element_size()
                )
                dst_ptrs[(src_dtype, src_block_size)].append(
                    exec.buffers[dst_block.buffer_type].data_ptr()
                )
                per_token_shape = exec.buffer_per_token_shape[
                    src_block.buffer_type
                ]
                shape_numel = int(np.prod(per_token_shape))
                n_elems[(src_dtype, src_block_size)].append(
                    src_n_tokens * shape_numel
                )
            src_offsets[(src_dtype, src_block_size)].append(0)
            dst_offsets[(src_dtype, src_block_size)].append(dst_block.index)
            src_strides[(src_dtype, src_block_size)].append(1)
            dst_strides[(src_dtype, src_block_size)].append(
                exec.buffer_block_strides[dst_block.buffer_type]
            )

    args_cache = {
        "other": {},
        "lse": {},
    }
    for dtype, block_size in src_ptrs.keys():
        src_ptr_tensor = torch.tensor(
            src_ptrs[(dtype, block_size)], dtype=torch.int64, device="cuda"
        )
        src_offsets_tensor = torch.tensor(
            src_offsets[(dtype, block_size)], dtype=torch.int32, device="cuda"
        )
        src_strides_tensor = torch.tensor(
            src_strides[(dtype, block_size)], dtype=torch.int32, device="cuda"
        )
        dst_ptr_tensor = torch.tensor(
            dst_ptrs[(dtype, block_size)], dtype=torch.int64, device="cuda"
        )
        dst_offsets_tensor = torch.tensor(
            dst_offsets[(dtype, block_size)], dtype=torch.int32, device="cuda"
        )
        dst_strides_tensor = torch.tensor(
            dst_strides[(dtype, block_size)], dtype=torch.int32, device="cuda"
        )
        n_elems_tensor = torch.tensor(
            n_elems[(dtype, block_size)], dtype=torch.int32, device="cuda"
        )
        args_cache["other"][(dtype, block_size)] = (
            src_ptr_tensor,
            src_offsets_tensor,
            src_strides_tensor,
            dst_ptr_tensor,
            dst_offsets_tensor,
            dst_strides_tensor,
            n_elems_tensor,
        )
    for dtype, block_size in lse_src_ptrs.keys():
        args_cache["lse"][(dtype, block_size)] = (
            torch.tensor(
                lse_src_ptrs[(dtype, block_size)],
                dtype=torch.int64,
                device="cuda",
            ),
            torch.tensor(
                lse_src_token_offsets[(dtype, block_size)],
                dtype=torch.int32,
                device="cuda",
            ),
            torch.tensor(
                lse_src_head_strides[(dtype, block_size)],
                dtype=torch.int32,
                device="cuda",
            ),
            torch.tensor(
                lse_dst_ptrs[(dtype, block_size)],
                dtype=torch.int64,
                device="cuda",
            ),
            torch.tensor(
                lse_dst_token_offsets[(dtype, block_size)],
                dtype=torch.int32,
                device="cuda",
            ),
            torch.tensor(
                lse_dst_head_strides[(dtype, block_size)],
                dtype=torch.int32,
                device="cuda",
            ),
            torch.tensor(
                lse_n_tokens[(dtype, block_size)],
                dtype=torch.int32,
                device="cuda",
            ),
            head_block_size,
        )

    exec._args_cache[instr_index] = args_cache
    torch.cuda.nvtx.range_pop()


def _handle_memcpy(exec: AttentionExecutor, instr: MemcpyInstr):
    torch.cuda.nvtx.range_push("MemcpyInstr")
    args_cache = exec._args_cache[exec.instr_index]

    for (dtype, block_size), args in args_cache["other"].items():
        if dtype == torch.float32:
            fused_copy_fp32_varlen(*args, block_size, 128)
        elif dtype == torch.float16:
            fused_copy_fp16_varlen(*args, block_size, 128)
        elif dtype == torch.bfloat16:
            fused_copy_bf16_varlen(*args, block_size, 128)
        else:
            raise ValueError(f"Unsupported dtype {dtype}")
    for (dtype, block_size), args in args_cache["lse"].items():
        if dtype == torch.float32:
            fused_copy_fp32_varlen_lse(*args, block_size, 128)
        else:
            raise ValueError(f"Unsupported dtype {dtype}")
    torch.cuda.nvtx.range_pop()


# def _handle_memcpy(exec: AttentionExecutor, instr: MemcpyInstr):
#     local_cu_seqlens = exec.execution_plan.local_cu_seqlens
#     for src_block, dst_block in instr.src_dst_pairs:
#         src_buffer_type = src_block.buffer_type
#         dst_buffer_type = dst_block.buffer_type
#         src_index = src_block.index
#         dst_index = dst_block.index
#         n_tokens = src_block.n_tokens
#         exec.logger.debug("Executing copy: MemcpyInstr: src_block: {}, dst_block: {}, local offset: {}, n_tokens: {}".format(src_block, dst_block, local_cu_seqlens[src_index], n_tokens))
#         if src_buffer_type in _LSE_BUFFER_TYPES or dst_buffer_type in _LSE_BUFFER_TYPES:
#             assert src_buffer_type in _LSE_BUFFER_TYPES and dst_buffer_type in _LSE_BUFFER_TYPES
#             exec.buffers[dst_buffer_type][:, dst_index, :n_tokens].copy_(
#                 exec.buffers[src_buffer_type][:, local_cu_seqlens[src_index] : local_cu_seqlens[src_index] + n_tokens]
#             )
#         else:
#             exec.buffers[dst_buffer_type][dst_index, :n_tokens].copy_(
#                 exec.buffers[src_buffer_type][local_cu_seqlens[src_index] : local_cu_seqlens[src_index] + n_tokens]
#             )


def _pad_block_table(table: List[List[BufferBlock]]) -> List[List[int]]:
    # extract buffer index from each block
    index_table = [[x.index for x in table_row] for table_row in table]
    max_n_blocks = max(len(table_row) for table_row in index_table)
    return [
        table_row + [-1] * (max_n_blocks - len(table_row))
        for table_row in index_table
    ]


def _prepare_attn(exec: AttentionExecutor, instr: AttnInstr):
    torch.cuda.nvtx.range_push("Prepare AttnInstr")
    cu_seqlens_q = torch.tensor(
        [0] + instr.seqlens_q, dtype=torch.int32, device="cuda"
    )
    cu_seqlens_q = torch.cumsum(cu_seqlens_q, dim=0, dtype=torch.int32)
    cu_seqlens_kv = torch.tensor(
        [0] + instr.seqlens_kv, dtype=torch.int32, device="cuda"
    )
    cu_seqlens_kv = torch.cumsum(cu_seqlens_kv, dim=0, dtype=torch.int32)

    total_q = sum(instr.seqlens_q)

    cu_q_block_table = torch.tensor(
        _pad_block_table(instr.q_block_table), dtype=torch.int32, device="cuda"
    )
    cu_kv_block_table = torch.tensor(
        _pad_block_table(instr.kv_block_table),
        dtype=torch.int32,
        device="cuda",
    )
    cu_out_block_table = torch.tensor(
        _pad_block_table(instr.out_block_table),
        dtype=torch.int32,
        device="cuda",
    )
    attn_mask = instr.attn_mask.to(torch.int32).cuda()

    exec._args_cache[exec.instr_index] = (
        total_q,
        cu_seqlens_q,
        cu_seqlens_kv,
        cu_q_block_table,
        cu_kv_block_table,
        cu_out_block_table,
        attn_mask,
    )
    torch.cuda.nvtx.range_pop()


def _handle_attn(exec: AttentionExecutor, instr: AttnInstr):
    torch.cuda.nvtx.range_push("AttnInstr")

    (
        total_q,
        cu_seqlens_q,
        cu_seqlens_kv,
        cu_q_block_table,
        cu_kv_block_table,
        cu_out_block_table,
        attn_mask,
    ) = exec._args_cache[exec.instr_index]

    # lse block table is the same as out
    _wrapped_flash_attn_varlen_forward(
        exec.buffers[BufferType.BUFFER_Q],
        exec.buffers[BufferType.BUFFER_KV],
        cu_seqlens_q,
        cu_seqlens_kv,
        total_q,
        instr.max_seqlen_q,
        instr.max_seqlen_kv,
        attn_mask=attn_mask,
        q_block_table=cu_q_block_table,
        kv_block_table=cu_kv_block_table,
        out_block_table=cu_out_block_table,
        out_=exec.buffers[BufferType.BUFFER_OUT],
        lse_=exec.buffers[BufferType.BUFFER_LSE],
    )
    torch.cuda.nvtx.range_pop()


def _prepare_attn_bw(exec: AttentionExecutor, instr: AttnBackwardInstr):
    torch.cuda.nvtx.range_push("Prepare AttnBackwardInstr")
    cu_seqlens_q = torch.tensor(
        [0] + instr.seqlens_q, dtype=torch.int32, device="cuda"
    )
    cu_seqlens_q = torch.cumsum(cu_seqlens_q, dim=0, dtype=torch.int32)
    cu_seqlens_kv = torch.tensor(
        [0] + instr.seqlens_kv, dtype=torch.int32, device="cuda"
    )
    cu_seqlens_kv = torch.cumsum(cu_seqlens_kv, dim=0, dtype=torch.int32)

    total_q = sum(instr.seqlens_q)
    total_kv = sum(instr.seqlens_kv)

    cu_q_block_table = torch.tensor(
        _pad_block_table(instr.q_block_table), dtype=torch.int32, device="cuda"
    )
    cu_kv_block_table = torch.tensor(
        _pad_block_table(instr.kv_block_table),
        dtype=torch.int32,
        device="cuda",
    )
    cu_out_block_table = torch.tensor(
        _pad_block_table(instr.out_block_table),
        dtype=torch.int32,
        device="cuda",
    )
    cu_dq_block_table = torch.tensor(
        _pad_block_table(instr.dq_block_table),
        dtype=torch.int32,
        device="cuda",
    )
    attn_mask = instr.attn_mask.to(torch.int32).cuda()

    is_gqa = (
        exec.buffer_per_token_shape[BufferType.BUFFER_dKV][-2]
        != exec.buffer_per_token_shape[BufferType.BUFFER_dQ][-2]
    )
    if is_gqa:
        # for GQA backward, we need to create a temporary dedicated dkv buffer
        # (and rewrite the block_table)
        total_dkv_blocks = sum(len(x) for x in instr.dkv_block_table)
        block_size = exec.buffer_block_sizes[BufferType.BUFFER_dKV]
        dkv_buffer = torch.zeros(
            total_dkv_blocks,
            block_size,
            2,
            exec.buffer_per_token_shape[BufferType.BUFFER_dKV][-2],
            exec.buffer_per_token_shape[BufferType.BUFFER_dKV][-1],
            dtype=exec.buffers[BufferType.BUFFER_dKV].dtype,
            device="cuda",
        )
        tmp_dkv_block_table = []
        dkv_cpy_src_ptrs = []
        dkv_cpy_dst_ptrs = []
        dkv_cpy_n_tokens = []
        dkv_cpy_max_n_tokens = exec.buffer_block_sizes[BufferType.BUFFER_dKV]
        dkv_cpy_per_token_size = int(
            np.prod(exec.buffer_per_token_shape[BufferType.BUFFER_dKV])
        )
        dkv_cpy_triton_block_size = exec.buffer_per_token_shape[
            BufferType.BUFFER_dKV
        ][-1]
        curr_n_blocks = 0
        for block_table_row in instr.dkv_block_table:
            tmp_dkv_block_table.append(
                list(
                    range(curr_n_blocks, curr_n_blocks + len(block_table_row))
                )
            )
            for i, block in enumerate(block_table_row):
                src_block_index = curr_n_blocks + i
                dst_block_index = block.index
                src_ptr = dkv_buffer[src_block_index].data_ptr()
                dst_ptr = exec.buffers[BufferType.BUFFER_dKV][
                    dst_block_index
                ].data_ptr()
                n_tokens = block.n_tokens
                dkv_cpy_src_ptrs.append(src_ptr)
                dkv_cpy_dst_ptrs.append(dst_ptr)
                dkv_cpy_n_tokens.append(n_tokens)
            curr_n_blocks += len(block_table_row)
        # manually pad tmp_dkv_block_table
        max_n_blocks = max(len(x) for x in tmp_dkv_block_table)
        tmp_dkv_block_table = [
            table_row + [-1] * (max_n_blocks - len(table_row))
            for table_row in tmp_dkv_block_table
        ]
        cu_dkv_block_table = torch.tensor(
            tmp_dkv_block_table,
            dtype=torch.int32,
            device="cuda",
        )
        # create block copy ops to copy data back to the main buffer
        dkv_cpy_src_ptrs_tensor = torch.tensor(
            dkv_cpy_src_ptrs,
            dtype=torch.int64,
            device="cuda",
        )
        dkv_cpy_dst_ptrs_tensor = torch.tensor(
            dkv_cpy_dst_ptrs,
            dtype=torch.int64,
            device="cuda",
        )
        dkv_cpy_n_tokens_tensor = torch.tensor(
            dkv_cpy_n_tokens,
            dtype=torch.int32,
            device="cuda",
        )
        exec._gqa_dkv_args[exec.instr_index] = (
            dkv_buffer,
            (
                dkv_cpy_src_ptrs_tensor,
                dkv_cpy_dst_ptrs_tensor,
                dkv_cpy_n_tokens_tensor,
                dkv_cpy_max_n_tokens,
                dkv_cpy_per_token_size,
                dkv_cpy_triton_block_size,
            ),
        )
    else:
        cu_dkv_block_table = torch.tensor(
            _pad_block_table(instr.dkv_block_table),
            dtype=torch.int32,
            device="cuda",
        )

    exec._args_cache[exec.instr_index] = (
        total_q,
        total_kv,
        cu_seqlens_q,
        cu_seqlens_kv,
        cu_q_block_table,
        cu_kv_block_table,
        cu_out_block_table,
        attn_mask,
        cu_dq_block_table,
        cu_dkv_block_table,
    )
    torch.cuda.nvtx.range_pop()


def _handle_attn_bw(exec: AttentionExecutor, instr: AttnBackwardInstr):
    torch.cuda.nvtx.range_push("AttnBackwardInstr")

    (
        total_q,
        total_kv,
        cu_seqlens_q,
        cu_seqlens_kv,
        cu_q_block_table,
        cu_kv_block_table,
        cu_out_block_table,
        attn_mask,
        cu_dq_block_table,
        cu_dkv_block_table,
    ) = exec._args_cache[exec.instr_index]

    if exec.instr_index in exec._gqa_dkv_args:
        dkv_buffer, dkv_cpy_args = exec._gqa_dkv_args[exec.instr_index]
        dkv_buffer.zero_()
    else:
        dkv_buffer = exec.buffers[BufferType.BUFFER_dKV]

    _wrapped_flash_attn_varlen_backward(
        exec.buffers[BufferType.BUFFER_dOUT],
        exec.buffers[BufferType.BUFFER_Q],
        exec.buffers[BufferType.BUFFER_KV],
        exec.buffers[BufferType.BUFFER_OUT],
        exec.buffers[BufferType.BUFFER_LSE],
        cu_seqlens_q,
        cu_seqlens_kv,
        total_q,
        total_kv,
        instr.max_seqlen_q,
        instr.max_seqlen_kv,
        attn_mask=attn_mask,
        q_block_table=cu_q_block_table,
        kv_block_table=cu_kv_block_table,
        out_block_table=cu_out_block_table,
        dq_block_table=cu_dq_block_table,
        dkv_block_table=cu_dkv_block_table,
        dq=exec.buffers[BufferType.BUFFER_dQ],
        dkv=dkv_buffer,
    )

    if exec.instr_index in exec._gqa_dkv_args:
        dtype = dkv_buffer.dtype
        if dtype == torch.float32:
            copy_fn = fused_copy_fp32_varlen_dkv
        elif dtype == torch.bfloat16:
            copy_fn = fused_copy_bf16_varlen_dkv
        elif dtype == torch.float16:
            copy_fn = fused_copy_fp16_varlen_dkv
        else:
            raise ValueError(f"Unsupported dtype {DType.as_str(dtype)}")
        copy_fn(
            *dkv_cpy_args,
        )
    torch.cuda.nvtx.range_pop()


def _round_to_pow_2(x: int):
    return 1 << (x - 1).bit_length()


def _prepare_reduction(exec: AttentionExecutor, instr: AttnReductionInstr):
    torch.cuda.nvtx.range_push("Prepare AttnReductionInstr")
    src_out_buffer_type = None
    src_lse_buffer_type = None
    dst_out_buffer_type = None
    dst_lse_buffer_type = None
    src_block_table = []
    dst_token_offsets = []
    n_tokens_per_block = []
    n_src_blocks_per_op = []
    output_is_unpadded = bool(instr.output_is_unpadded)
    for red_op in instr.ops:
        # lse block index is always the same as out block index
        src_out_buffer_type = red_op.src_buffers[0][0].buffer_type
        src_lse_buffer_type = red_op.src_buffers[0][1].buffer_type
        dst_out_buffer_type = red_op.dst_buffer[0].buffer_type
        dst_lse_buffer_type = red_op.dst_buffer[1].buffer_type
        src_block_indices = [x[0].index for x in red_op.src_buffers]
        src_block_table.append(src_block_indices)
        if output_is_unpadded:
            # dst out buffer is unpadded, calculate offset from local_cu_seqlens
            dst_block_index = red_op.dst_buffer[0].index
            dst_token_offsets.append(
                exec.execution_plan.local_cu_seqlens[dst_block_index]
            )
        else:
            # dst output buffer is padded, calculate offset from buffer index
            dst_block_index = red_op.dst_buffer[0].index
            dst_token_offsets.append(
                dst_block_index * exec.buffer_block_sizes[dst_out_buffer_type]
            )
        n_tokens_per_block.append(red_op.dst_buffer[0].n_tokens)
        n_src_blocks_per_op.append(len(src_block_indices))
    # pad src block table
    max_n_src_blocks = _round_to_pow_2(max(len(x) for x in src_block_table))

    src_block_table = [
        x + [-1] * (max_n_src_blocks - len(x)) for x in src_block_table
    ]
    src_block_table = torch.tensor(
        src_block_table, dtype=torch.int32, device="cuda"
    )
    n_src_blocks_per_op = torch.tensor(
        n_src_blocks_per_op, dtype=torch.int32, device="cuda"
    )
    dst_token_offsets = torch.tensor(
        dst_token_offsets, dtype=torch.int32, device="cuda"
    )
    n_total_tokens = sum(n_tokens_per_block)
    n_tokens_per_block = torch.tensor(
        n_tokens_per_block, dtype=torch.int32, device="cuda"
    )
    if exec.logger:
        exec.logger.debug(
            "AttnReductionInstr: max_n_src_blocks: {} , head_dim: {}".format(
                max_n_src_blocks, exec.head_dim
            )
        )
    exec._args_cache[exec.instr_index] = (
        src_out_buffer_type,
        src_lse_buffer_type,
        dst_out_buffer_type,
        dst_lse_buffer_type,
        src_block_table,
        dst_token_offsets,
        n_tokens_per_block,
        n_src_blocks_per_op,
        max_n_src_blocks,
        n_total_tokens,
    )
    torch.cuda.nvtx.range_pop()


# reference implementation from
# https://github.com/zhuzilin/ring-flash-attention/blob/main/ring_flash_attn/utils.py
@torch.jit.script
def _update_out_and_lse(
    out: torch.Tensor,
    lse: torch.Tensor,
    block_out: torch.Tensor,
    block_lse: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:

    block_out = block_out.to(torch.float32)
    block_lse = block_lse.transpose(-2, -1).unsqueeze(dim=-1)

    # new_lse = lse + torch.log(1 + torch.exp(block_lse - lse))
    # torch.exp(lse - new_lse) * out + torch.exp(block_lse - new_lse) * block_out
    # For additional context and discussion, please refer to:
    # https://github.com/zhuzilin/ring-flash-attention/pull/34#issuecomment-2076126795
    out = out - F.sigmoid(block_lse - lse) * (out - block_out)
    lse = lse - F.logsigmoid(lse - block_lse)

    return out, lse


# reference implementation from
# https://github.com/zhuzilin/ring-flash-attention/blob/main/ring_flash_attn/utils.py
def update_out_and_lse(
    out: Optional[torch.Tensor],
    lse: Optional[torch.Tensor],
    block_out: torch.Tensor,
    block_lse: torch.Tensor,
    slice_=None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if out is None:
        if slice_ is not None:
            raise RuntimeError(
                "first update_out_and_lse should not pass slice_ args"
            )
        out = block_out.to(torch.float32)
        lse = block_lse.transpose(-2, -1).unsqueeze(dim=-1)
    elif slice_ is not None:
        slice_out, slice_lse = out[slice_], lse[slice_]
        slice_out, slice_lse = _update_out_and_lse(
            slice_out, slice_lse, block_out, block_lse
        )
        out[slice_], lse[slice_] = slice_out, slice_lse
    else:
        out, lse = _update_out_and_lse(out, lse, block_out, block_lse)
    return out, lse


def _reference_reduction(exec: AttentionExecutor, instr: AttnReductionInstr):
    result_outs = []
    result_lses = []
    for red_op in instr.ops:
        src_out = None
        src_lse = None
        n_tokens = red_op.dst_buffer[0].n_tokens
        for src_out_block, src_lse_block in red_op.src_buffers:
            src_out_tensor = exec.buffers[src_out_block.buffer_type][
                src_out_block.index, :n_tokens
            ].clone()
            src_lse_tensor = exec.buffers[src_lse_block.buffer_type][
                :, src_lse_block.index, :n_tokens
            ].clone()
            src_out, src_lse = update_out_and_lse(
                src_out, src_lse, src_out_tensor, src_lse_tensor
            )
        ref_out = src_out
        result_lse = src_lse.squeeze(-1).transpose(-2, -1)
        result_outs.append(ref_out.clone())
        result_lses.append(result_lse.clone())
    return result_outs, result_lses


def _handle_reduction(exec: AttentionExecutor, instr: AttnReductionInstr):
    torch.cuda.nvtx.range_push("AttnReductionInstr")

    (
        src_out_buffer_type,
        src_lse_buffer_type,
        dst_out_buffer_type,
        dst_lse_buffer_type,
        src_block_table,
        dst_token_offsets,
        n_tokens_per_block,
        n_src_blocks_per_op,
        max_n_src_blocks,
        n_total_tokens,
    ) = exec._args_cache[exec.instr_index]

    if DEBUG_LOG_REDUCTION_ARGS_ON_ERROR:
        # we run reference first since triton kernel may overwrite the input
        result_outs, result_lses = _reference_reduction(exec, instr)
        # for op_id, red_op in enumerate(instr.ops):
        #     dst_out_block, dst_lse_block = red_op.dst_buffer
        #     ref_out = result_outs[op_id]
        #     ref_lse = result_lses[op_id]
        #     exec.buffers[dst_out_buffer_type][dst_out_block.index].copy_(
        #         ref_out
        #     )
        #     exec.buffers[dst_lse_block.buffer_type][
        #         :, dst_lse_block.index
        #     ].copy_(ref_lse)

    try:
        flash_attn_update_out_lse(
            exec.buffers[src_out_buffer_type],
            exec.buffers[dst_out_buffer_type],
            exec.buffers[src_lse_buffer_type],
            exec.buffers[dst_lse_buffer_type],
            exec.n_heads_per_block,
            src_block_table,
            dst_token_offsets,
            n_tokens_per_block,
            n_src_blocks_per_op,
            exec.buffer_block_sizes[src_out_buffer_type],
            exec.buffer_block_strides[src_out_buffer_type],
            exec.n_heads_per_block * exec.head_dim,
            exec.buffer_block_sizes[src_lse_buffer_type],
            1,
            exec.buffer_n_blocks[src_lse_buffer_type]
            * exec.buffer_block_sizes[src_lse_buffer_type],
            (
                n_total_tokens
                if instr.output_is_unpadded
                else (
                    exec.buffer_n_blocks[src_lse_buffer_type]
                    * exec.buffer_block_sizes[src_lse_buffer_type]
                )
            ),
            exec.head_dim,
            max_n_src_blocks,
            32 if IS_L40 else 64,
            # require_cached=True,
        )
    except Exception as e:
        exec.logger.error(str(e))
        raise e

    # local_cu_seqlens = exec.execution_plan.local_cu_seqlens
    # result_outs, result_lses = _reference_reduction(exec, instr)
    # for op_id, red_op in enumerate(instr.ops):
    #     dst_out_block, dst_lse_block = red_op.dst_buffer
    #     ref_out = result_outs[op_id]
    #     ref_lse = result_lses[op_id]
    #     n_tokens = dst_out_block.n_tokens
    #     if instr.output_is_unpadded:
    #         dst_token_offset = local_cu_seqlens[dst_out_block.index]
    #         exec.buffers[dst_out_buffer_type][dst_token_offset: dst_token_offset + n_tokens].copy_(
    #             ref_out
    #         )
    #         exec.buffers[dst_lse_block.buffer_type][
    #             :, dst_token_offset: dst_token_offset + n_tokens
    #         ].copy_(ref_lse)
    #     else:
    #         exec.buffers[dst_out_buffer_type][dst_out_block.index, :n_tokens].copy_(
    #             ref_out
    #         )
    #         exec.buffers[dst_lse_block.buffer_type][
    #             :, dst_out_block.index, :n_tokens
    #         ].copy_(ref_lse)

    if DEBUG_LOG_REDUCTION_ARGS_ON_ERROR:
        for op_id, red_op in enumerate(instr.ops):
            _, dst_lse_block = red_op.dst_buffer
            ref_lse = result_lses[op_id]
            token_offset = exec.execution_plan.local_cu_seqlens[
                red_op.dst_buffer[0].index
            ]
            if not torch.allclose(
                ref_lse,
                exec.buffers[dst_lse_block.buffer_type][
                    :,
                    token_offset : token_offset
                    + red_op.dst_buffer[0].n_tokens,
                ],
                rtol=1e-4,
            ):
                # generate random file name
                import uuid
                from pathlib import Path

                dirpath = Path.home() / ".dcp_debug"
                if not dirpath.exists():
                    dirpath.mkdir()
                filename = str(dirpath / ("LSE-" + str(uuid.uuid4()) + ".pt"))
                # save arguments for debugging
                torch.save(
                    {
                        "triton_args": (
                            exec.buffers[src_out_buffer_type],
                            exec.buffers[dst_out_buffer_type],
                            exec.buffers[src_lse_buffer_type],
                            exec.buffers[dst_lse_buffer_type],
                            exec.n_heads_per_block,
                            src_block_table,
                            dst_token_offsets,
                            n_tokens_per_block,
                            n_src_blocks_per_op,
                            exec.buffer_block_sizes[src_out_buffer_type],
                            exec.buffer_block_strides[src_out_buffer_type],
                            exec.n_heads_per_block * exec.head_dim,
                            exec.buffer_block_sizes[src_lse_buffer_type],
                            1,
                            exec.buffer_n_blocks[src_lse_buffer_type]
                            * exec.buffer_block_sizes[src_lse_buffer_type],
                            n_total_tokens,
                            exec.head_dim,
                            max_n_src_blocks,
                            64,
                        ),
                        "torch_args": (red_op),
                    },
                    filename,
                )

    torch.cuda.nvtx.range_pop()


def _prepare_sum(exec: AttentionExecutor, instr: SumInstr):
    torch.cuda.nvtx.range_push("Prepare SumInstr")
    src_ptrs = defaultdict(list)
    src_offsets = defaultdict(list)
    src_strides = defaultdict(list)
    n_ops_per_sums = defaultdict(list)
    dst_ptrs = defaultdict(list)
    dst_offsets = defaultdict(list)
    dst_strides = defaultdict(list)
    n_elems = defaultdict(list)
    max_n_ops_per_sums = defaultdict(int)
    for sum_op in instr.ops:
        src_buffer_blocks = sum_op.src_buffers
        dst_buffer_block = sum_op.dst_buffer
        src_buffer_type = src_buffer_blocks[0].buffer_type
        dst_buffer_type = dst_buffer_block.buffer_type
        dtype = exec.buffers[src_buffer_type].dtype
        block_size = exec.buffer_block_strides[src_buffer_type]
        src_buffer_stride = exec.buffer_block_strides[src_buffer_type]
        src_offset = [x.index for x in src_buffer_blocks]
        n_ops_per_sum = len(src_buffer_blocks)
        per_token_numel = int(
            np.prod(exec.buffer_per_token_shape[src_buffer_type])
        )
        n_elem = dst_buffer_block.n_tokens * per_token_numel
        if instr.output_is_unpadded:
            local_cu_seqlens = exec.execution_plan.local_cu_seqlens
            dst_buffer_offset = local_cu_seqlens[dst_buffer_block.index]
            dst_buffer_stride = per_token_numel
        else:
            dst_buffer_offset = dst_buffer_block.index
            dst_buffer_stride = exec.buffer_block_strides[dst_buffer_type]
        src_ptrs[(dtype, block_size)].append(
            exec.buffers[src_buffer_type].data_ptr()
        )
        src_offsets[(dtype, block_size)].append(src_offset)
        src_strides[(dtype, block_size)].append(src_buffer_stride)
        n_ops_per_sums[(dtype, block_size)].append(n_ops_per_sum)
        dst_ptrs[(dtype, block_size)].append(
            exec.buffers[dst_buffer_type].data_ptr()
        )
        dst_offsets[(dtype, block_size)].append(dst_buffer_offset)
        dst_strides[(dtype, block_size)].append(dst_buffer_stride)
        n_elems[(dtype, block_size)].append(n_elem)
    for dtype, block_size in src_ptrs.keys():
        max_n_ops = max(n_ops_per_sums[(dtype, block_size)])
        max_n_ops = _round_to_pow_2(max_n_ops)
        src_ptrs[(dtype, block_size)] = torch.tensor(
            src_ptrs[(dtype, block_size)], dtype=torch.int64, device="cuda"
        )
        padded_src_offsets = []
        for src_offsets_list in src_offsets[(dtype, block_size)]:
            padded_src_offsets.append(
                src_offsets_list + [-1] * (max_n_ops - len(src_offsets_list))
            )
        src_offsets[(dtype, block_size)] = torch.tensor(
            padded_src_offsets, dtype=torch.int32, device="cuda"
        )
        src_strides[(dtype, block_size)] = torch.tensor(
            src_strides[(dtype, block_size)], dtype=torch.int32, device="cuda"
        )
        n_ops_per_sums[(dtype, block_size)] = torch.tensor(
            n_ops_per_sums[(dtype, block_size)],
            dtype=torch.int32,
            device="cuda",
        )
        dst_ptrs[(dtype, block_size)] = torch.tensor(
            dst_ptrs[(dtype, block_size)], dtype=torch.int64, device="cuda"
        )
        dst_offsets[(dtype, block_size)] = torch.tensor(
            dst_offsets[(dtype, block_size)], dtype=torch.int32, device="cuda"
        )
        dst_strides[(dtype, block_size)] = torch.tensor(
            dst_strides[(dtype, block_size)], dtype=torch.int32, device="cuda"
        )
        n_elems[(dtype, block_size)] = torch.tensor(
            n_elems[(dtype, block_size)], dtype=torch.int32, device="cuda"
        )
        max_n_ops_per_sums[(dtype, block_size)] = max_n_ops
    if exec.logger:
        for (
            dtype,
            block_size,
        ), max_n_ops_per_sum in max_n_ops_per_sums.items():
            exec.logger.debug(
                "SumInstr: dtype: {}, block_size: {}, max_n_ops_per_sum: {}".format(
                    dtype, block_size, max_n_ops_per_sum
                )
            )
    exec._args_cache[exec.instr_index] = (
        src_ptrs,
        src_offsets,
        src_strides,
        n_ops_per_sums,
        dst_ptrs,
        dst_offsets,
        dst_strides,
        n_elems,
        max_n_ops_per_sums,
    )
    torch.cuda.nvtx.range_pop()


def _handle_sum(exec: AttentionExecutor, instr: SumInstr):
    torch.cuda.nvtx.range_push("SumInstr")
    (
        src_ptrs,
        src_offsets,
        src_strides,
        n_ops_per_sums,
        dst_ptrs,
        dst_offsets,
        dst_strides,
        n_elems,
        max_n_ops_per_sums,
    ) = exec._args_cache[exec.instr_index]
    for dtype, block_size in src_ptrs.keys():
        fused_blockwise_sum(
            src_ptrs[(dtype, block_size)],
            src_offsets[(dtype, block_size)],
            src_strides[(dtype, block_size)],
            n_ops_per_sums[(dtype, block_size)],
            dst_ptrs[(dtype, block_size)],
            dst_offsets[(dtype, block_size)],
            dst_strides[(dtype, block_size)],
            n_elems[(dtype, block_size)],
            dtype,
            max_n_ops_per_sums[(dtype, block_size)],
            block_size,
            128,
            # require_cached=True,
        )
    torch.cuda.nvtx.range_pop()


# def _handle_sum(exec: AttentionExecutor, instr: SumInstr):
#     torch.cuda.nvtx.range_push("SumInstr")
#     for sum_op in instr.ops:
#         src_buffer_blocks = sum_op.src_buffers
#         dst_buffer_block = sum_op.dst_buffer
#         src_buffer_type = src_buffer_blocks[0].buffer_type
#         dst_buffer_type = dst_buffer_block.buffer_type
#         src_buffer_indices = [x.index for x in src_buffer_blocks]
#         dst_buffer_index = dst_buffer_block.index
#         n_tokens = dst_buffer_block.n_tokens
#         torch.cuda.nvtx.range_push("src_buffer Index")
#         src_buffer = exec.buffers[src_buffer_type][src_buffer_indices]
#         torch.cuda.nvtx.range_pop()
#         torch.cuda.nvtx.range_push("dst_buffer Index")
#         if instr.output_is_unpadded:
#             src_buffer = src_buffer[:, :n_tokens]
#             local_cu_seqlens = exec.execution_plan.local_cu_seqlens
#             dst_buffer = exec.buffers[dst_buffer_type][
#                 local_cu_seqlens[dst_buffer_index] : local_cu_seqlens[
#                     dst_buffer_index
#                 ]
#                 + dst_buffer_block.n_tokens
#             ]
#         else:
#             dst_buffer = exec.buffers[dst_buffer_type][dst_buffer_index]
#         torch.cuda.nvtx.range_pop()
#         # if src buffer contains nan, replace it with 0
#         src_buffer.nan_to_num_(0.0)
#         dst_buffer.copy_(
#             src_buffer.sum(dim=0, dtype=torch.float32).to(dst_buffer.dtype)
#         )
#     torch.cuda.nvtx.range_pop()


def _prepare_comm_launch(exec: AttentionExecutor, instr: CommLaunchInstr):
    torch.cuda.nvtx.range_push("Prepare CommLaunchInstr")
    if exec.logger:
        exec.logger.debug("Preparing {}".format(instr))
    if instr.stream is not None:
        if instr.stream not in exec.streams:
            exec.streams[instr.stream] = torch.cuda.Stream()

    send_src_blocks = defaultdict(list)
    recv_dst_blocks = defaultdict(list)
    send_buffers = {}
    recv_buffers = {}
    send_views = defaultdict(list)
    recv_views = defaultdict(list)
    # try use a single kernel launch for send/recv copying
    send_cpy_src_ptrs = defaultdict(list)
    send_cpy_dst_ptrs = defaultdict(list)
    send_cpy_src_offsets = defaultdict(list)
    send_cpy_dst_offsets = defaultdict(list)
    send_cpy_src_strides = defaultdict(list)
    send_cpy_dst_strides = defaultdict(list)
    send_cpy_dst_n_elems = defaultdict(list)

    recv_cpy_src_ptrs = defaultdict(list)
    recv_cpy_dst_ptrs = defaultdict(list)
    recv_cpy_src_offsets = defaultdict(list)
    recv_cpy_dst_offsets = defaultdict(list)
    recv_cpy_src_strides = defaultdict(list)
    recv_cpy_dst_strides = defaultdict(list)
    recv_cpy_dst_n_elems = defaultdict(list)
    unique_dtypes = set()
    torch.cuda.nvtx.range_push("Prepare Send")
    for comm_op in instr.comm_ops:
        peer = exec.get_rank(*comm_op.peer)
        buffer_type = comm_op.buffer_block.buffer_type
        buffer_block_numel = exec.buffer_block_strides[buffer_type]
        buffer_index = comm_op.buffer_block.index
        n_tokens = comm_op.buffer_block.n_tokens
        if buffer_type in _LOCAL_UNPADDED_BUFFER_TYPES:
            token_offset = exec.execution_plan.local_cu_seqlens[buffer_index]
            if buffer_type not in _LSE_BUFFER_TYPES:
                buffer_block = exec.buffers[buffer_type][
                    token_offset : token_offset + n_tokens
                ]
            else:
                buffer_block = exec.buffers[buffer_type][
                    :, token_offset : token_offset + n_tokens
                ]
        else:
            if buffer_type not in _LSE_BUFFER_TYPES:
                buffer_block = exec.buffers[buffer_type][
                    buffer_index, :n_tokens
                ]
            else:
                buffer_block = exec.buffers[buffer_type][
                    :, buffer_index, :n_tokens
                ]
        dtype = torch_dtype_to_int(buffer_block.dtype)
        if comm_op.comm_type == CommType.SEND:
            send_src_blocks[(peer, dtype)].append(
                (buffer_block, buffer_block_numel)
            )
        else:
            recv_dst_blocks[(peer, dtype)].append(
                (buffer_block, buffer_block_numel)
            )
    # to use all-to-all, we need a single send buffer for all peers
    send_buffers = {}  # key = dtype
    send_numels_per_dtype_per_peer = defaultdict(dict)
    for (peer, dtype), send_buffers_and_block_sizes in send_src_blocks.items():
        send_buffer_list, buffer_block_sizes = zip(
            *send_buffers_and_block_sizes
        )
        send_buffer_list: List[torch.Tensor]
        send_numels = [t.numel() for t in send_buffer_list]
        send_numels_per_dtype_per_peer[dtype][peer] = send_numels
        unique_dtypes.add(dtype)
    send_buffer_peer_cumsum_offsets = defaultdict(dict)
    send_buffer_peer_sum_numel = defaultdict(dict)
    for dtype, numels_per_peer in send_numels_per_dtype_per_peer.items():
        total_buffer_size = 0
        sorted_peers = sorted(list(numels_per_peer.keys()))
        for peer in sorted_peers:
            peer_numels = numels_per_peer[peer]
            sum_peer_numel = sum(peer_numels)
            send_buffer_peer_sum_numel[dtype][peer] = sum_peer_numel
            cumsum = np.cumsum([total_buffer_size] + peer_numels).tolist()
            send_buffer_peer_cumsum_offsets[dtype][peer] = cumsum
            total_buffer_size += sum_peer_numel
        send_buffers[dtype] = torch.zeros(
            total_buffer_size, dtype=int_to_torch_dtype(dtype), device="cuda"
        )
    # we need a separate copy kernel launch for each dtype and
    # size combination (across peers).
    for (peer, dtype), send_buffers_and_block_sizes in send_src_blocks.items():
        send_buffer_list, buffer_block_sizes = zip(
            *send_buffers_and_block_sizes
        )
        send_buffer_list: List[torch.Tensor]
        send_buffer = send_buffers[dtype]
        cumsum_send_numels = send_buffer_peer_cumsum_offsets[dtype][peer]
        if exec.logger:
            send_numel = cumsum_send_numels[-1] - cumsum_send_numels[0]
            exec.logger.debug(
                "Sending {} bytes (dtype {}) to peer {}".format(
                    send_numel * send_buffer.element_size(),
                    int_to_torch_dtype(dtype),
                    exec.reverse_get_rank(peer),
                )
            )

        if not send_buffer_list[0].is_contiguous():
            # cannot use custom kernels, fallback to pytorch copy
            for i in range(len(send_buffer_list)):
                send_view = send_buffer[
                    cumsum_send_numels[i] : cumsum_send_numels[i + 1]
                ]
                src_view = send_buffer_list[i]
                send_views[dtype].append(
                    (send_view.view(src_view.shape), src_view)
                )
            continue

        send_src_ptrs = [t.data_ptr() for t in send_buffer_list]
        send_buffer_ptrs = [send_buffer.data_ptr() for _ in send_buffer_list]
        send_buffer_offsets = cumsum_send_numels[:-1]
        # data_ptr of tensor views already points to the correct offset
        send_src_offsets = [0] * len(send_buffer_list)
        send_buffer_strides = [1] * len(send_buffer_list)
        send_src_strides = [1] * len(send_buffer_list)
        send_n_elems = [t.numel() for t in send_buffer_list]

        for i in range(len(send_buffer_list)):
            key = (dtype, buffer_block_sizes[i])
            send_cpy_src_ptrs[key].append(send_src_ptrs[i])
            send_cpy_dst_ptrs[key].append(send_buffer_ptrs[i])
            send_cpy_src_offsets[key].append(send_src_offsets[i])
            send_cpy_dst_offsets[key].append(send_buffer_offsets[i])
            send_cpy_src_strides[key].append(send_src_strides[i])
            send_cpy_dst_strides[key].append(send_buffer_strides[i])
            send_cpy_dst_n_elems[key].append(send_n_elems[i])

    send_cpy_src_ptrs_tensors = {
        key: torch.tensor(val, dtype=torch.int64, device="cuda")
        for key, val in send_cpy_src_ptrs.items()
    }
    send_cpy_dst_ptrs_tensors = {
        key: torch.tensor(val, dtype=torch.int64, device="cuda")
        for key, val in send_cpy_dst_ptrs.items()
    }
    send_cpy_src_offsets_tensors = {
        key: torch.tensor(val, dtype=torch.int32, device="cuda")
        for key, val in send_cpy_src_offsets.items()
    }
    send_cpy_dst_offsets_tensors = {
        key: torch.tensor(val, dtype=torch.int32, device="cuda")
        for key, val in send_cpy_dst_offsets.items()
    }
    send_cpy_src_strides_tensors = {
        key: torch.tensor(val, dtype=torch.int32, device="cuda")
        for key, val in send_cpy_src_strides.items()
    }
    send_cpy_dst_strides_tensors = {
        key: torch.tensor(val, dtype=torch.int32, device="cuda")
        for key, val in send_cpy_dst_strides.items()
    }
    send_cpy_dst_n_elems_tensors = {
        key: torch.tensor(val, dtype=torch.int32, device="cuda")
        for key, val in send_cpy_dst_n_elems.items()
    }
    torch.cuda.nvtx.range_pop()
    torch.cuda.nvtx.range_push("Prepare Recv")
    recv_buffers = {}  # key = dtype
    recv_numels_per_dtype_per_peer = defaultdict(dict)
    for (peer, dtype), recv_buffers_and_block_sizes in recv_dst_blocks.items():
        recv_buffer_list, buffer_block_sizes = zip(
            *recv_buffers_and_block_sizes
        )
        recv_buffer_list: List[torch.Tensor]
        recv_numels = [t.numel() for t in recv_buffer_list]
        recv_numels_per_dtype_per_peer[dtype][peer] = recv_numels
        unique_dtypes.add(dtype)
    recv_buffer_peer_cumsum_offsets = defaultdict(dict)
    recv_buffer_peer_sum_numel = defaultdict(dict)
    for dtype, numels_per_peer in recv_numels_per_dtype_per_peer.items():
        total_buffer_size = 0
        sorted_peers = sorted(list(numels_per_peer.keys()))
        for peer in sorted_peers:
            peer_numels = numels_per_peer[peer]
            sum_peer_numel = sum(peer_numels)
            recv_buffer_peer_sum_numel[dtype][peer] = sum_peer_numel
            cumsum = np.cumsum([total_buffer_size] + peer_numels).tolist()
            recv_buffer_peer_cumsum_offsets[dtype][peer] = cumsum
            total_buffer_size += sum_peer_numel
        recv_buffers[dtype] = torch.zeros(
            total_buffer_size, dtype=int_to_torch_dtype(dtype), device="cuda"
        )
    # create dummy tensor for send only / recv only cases
    for dtype in unique_dtypes:
        if dtype not in send_buffers:
            send_buffers[dtype] = torch.empty(
                0, dtype=int_to_torch_dtype(dtype), device="cuda"
            )
        if dtype not in recv_buffers:
            recv_buffers[dtype] = torch.empty(
                0, dtype=int_to_torch_dtype(dtype), device="cuda"
            )
    # recv_buffer_blocks is organized by peer and dtype
    # here we organize by dtype and size
    for (peer, dtype), recv_buffers_and_block_sizes in recv_dst_blocks.items():
        recv_buffer_list, buffer_block_sizes = zip(
            *recv_buffers_and_block_sizes
        )
        recv_buffer_list: List[torch.Tensor]
        recv_buffer = recv_buffers[dtype]
        cumsum_recv_numels = recv_buffer_peer_cumsum_offsets[dtype][peer]
        if exec.logger:
            recv_numel = cumsum_recv_numels[-1] - cumsum_recv_numels[0]
            exec.logger.debug(
                "Recving {} bytes (dtype {}) from peer {}".format(
                    recv_numel * recv_buffer.element_size(),
                    int_to_torch_dtype(dtype),
                    exec.reverse_get_rank(peer),
                )
            )

        if not recv_buffer_list[0].is_contiguous():
            # cannot use custom kernels, fallback to pytorch copy
            for i in range(len(recv_buffer_list)):
                recv_view = recv_buffer[
                    cumsum_recv_numels[i] : cumsum_recv_numels[i + 1]
                ]
                target_view = recv_buffer_list[i]
                recv_views[dtype].append(
                    (recv_view.view(target_view.shape), target_view)
                )
            continue

        recv_buffer_ptrs = [recv_buffer.data_ptr() for _ in recv_buffer_list]
        recv_tgt_ptrs = [t.data_ptr() for t in recv_buffer_list]
        recv_buffer_offsets = cumsum_recv_numels[:-1]
        # data_ptr of tensor views already points to the correct offset
        recv_tgt_offsets = [0] * len(recv_buffer_list)
        recv_buffer_strides = [1] * len(recv_buffer_list)
        recv_tgt_strides = [1] * len(recv_buffer_list)
        recv_n_elems = [t.numel() for t in recv_buffer_list]

        for i in range(len(recv_buffer_list)):
            key = (dtype, buffer_block_sizes[i])
            recv_cpy_src_ptrs[key].append(recv_buffer_ptrs[i])
            recv_cpy_dst_ptrs[key].append(recv_tgt_ptrs[i])
            recv_cpy_src_offsets[key].append(recv_buffer_offsets[i])
            recv_cpy_dst_offsets[key].append(recv_tgt_offsets[i])
            recv_cpy_src_strides[key].append(recv_buffer_strides[i])
            recv_cpy_dst_strides[key].append(recv_tgt_strides[i])
            recv_cpy_dst_n_elems[key].append(recv_n_elems[i])

    recv_cpy_src_ptrs_tensors = {
        key: torch.tensor(val, dtype=torch.int64, device="cuda")
        for key, val in recv_cpy_src_ptrs.items()
    }
    recv_cpy_dst_ptrs_tensors = {
        key: torch.tensor(val, dtype=torch.int64, device="cuda")
        for key, val in recv_cpy_dst_ptrs.items()
    }
    recv_cpy_src_offsets_tensors = {
        key: torch.tensor(val, dtype=torch.int32, device="cuda")
        for key, val in recv_cpy_src_offsets.items()
    }
    recv_cpy_dst_offsets_tensors = {
        key: torch.tensor(val, dtype=torch.int32, device="cuda")
        for key, val in recv_cpy_dst_offsets.items()
    }
    recv_cpy_src_strides_tensors = {
        key: torch.tensor(val, dtype=torch.int32, device="cuda")
        for key, val in recv_cpy_src_strides.items()
    }
    recv_cpy_dst_strides_tensors = {
        key: torch.tensor(val, dtype=torch.int32, device="cuda")
        for key, val in recv_cpy_dst_strides.items()
    }
    recv_cpy_dst_n_elems_tensors = {
        key: torch.tensor(val, dtype=torch.int32, device="cuda")
        for key, val in recv_cpy_dst_n_elems.items()
    }
    torch.cuda.nvtx.range_pop()
    # create p2p ops
    output_split_sizes = {}
    for dtype in unique_dtypes:
        output_split_sizes[dtype] = []
        for peer in range(exec.n_devices_per_node * exec.n_nodes):
            if peer in recv_buffer_peer_sum_numel[dtype]:
                output_split_sizes[dtype].append(
                    recv_buffer_peer_sum_numel[dtype][peer]
                )
            else:
                output_split_sizes[dtype].append(0)
    input_split_sizes = {}
    for dtype in unique_dtypes:
        input_split_sizes[dtype] = []
        for peer in range(exec.n_devices_per_node * exec.n_nodes):
            if peer in send_buffer_peer_sum_numel[dtype]:
                input_split_sizes[dtype].append(
                    send_buffer_peer_sum_numel[dtype][peer]
                )
            else:
                input_split_sizes[dtype].append(0)
    exec._args_cache[exec.instr_index] = (
        send_buffers,
        send_views,
        recv_buffers,
        recv_views,
        input_split_sizes,
        output_split_sizes,
        send_cpy_src_ptrs_tensors,
        send_cpy_dst_ptrs_tensors,
        send_cpy_src_offsets_tensors,
        send_cpy_dst_offsets_tensors,
        send_cpy_src_strides_tensors,
        send_cpy_dst_strides_tensors,
        send_cpy_dst_n_elems_tensors,
        recv_cpy_src_ptrs_tensors,
        recv_cpy_dst_ptrs_tensors,
        recv_cpy_src_offsets_tensors,
        recv_cpy_dst_offsets_tensors,
        recv_cpy_src_strides_tensors,
        recv_cpy_dst_strides_tensors,
        recv_cpy_dst_n_elems_tensors,
    )
    torch.cuda.nvtx.range_pop()


def _handle_comm_launch(exec: AttentionExecutor, instr: CommLaunchInstr):
    torch.cuda.nvtx.range_push("CommLaunchInstr")
    (
        send_buffers,
        send_views,
        recv_buffers,
        recv_views,
        # p2p_ops_per_dtype,
        input_split_sizes,
        output_split_sizes,
        send_cpy_src_ptrs_tensors,
        send_cpy_dst_ptrs_tensors,
        send_cpy_src_offsets_tensors,
        send_cpy_dst_offsets_tensors,
        send_cpy_src_strides_tensors,
        send_cpy_dst_strides_tensors,
        send_cpy_dst_n_elems_tensors,
        recv_cpy_src_ptrs_tensors,
        recv_cpy_dst_ptrs_tensors,
        recv_cpy_src_offsets_tensors,
        recv_cpy_dst_offsets_tensors,
        recv_cpy_src_strides_tensors,
        recv_cpy_dst_strides_tensors,
        recv_cpy_dst_n_elems_tensors,
    ) = exec._args_cache[exec.instr_index]

    if instr.stream is not None:
        stream = exec.streams[instr.stream]
    else:
        stream = None

    with torch.cuda.stream(stream):
        # first copy data to send buffers
        # launch copy kernels
        for dtype, block_size in send_cpy_src_ptrs_tensors.keys():
            if dtype == DType.FLOAT32:
                copy_fn = fused_copy_fp32_varlen
            elif dtype == DType.BFLOAT16:
                copy_fn = fused_copy_bf16_varlen
            elif dtype == DType.FLOAT16:
                copy_fn = fused_copy_fp16_varlen
            else:
                raise ValueError(f"Unsupported dtype {DType.as_str(dtype)}")
            copy_fn(
                send_cpy_src_ptrs_tensors[(dtype, block_size)],
                send_cpy_src_offsets_tensors[(dtype, block_size)],
                send_cpy_src_strides_tensors[(dtype, block_size)],
                send_cpy_dst_ptrs_tensors[(dtype, block_size)],
                send_cpy_dst_offsets_tensors[(dtype, block_size)],
                send_cpy_dst_strides_tensors[(dtype, block_size)],
                send_cpy_dst_n_elems_tensors[(dtype, block_size)],
                block_size,
                128,
            )
        # handle pytorch fallbacks
        for dtype, views in send_views.items():
            for send_view, src_view in views:
                send_view.copy_(src_view)

        pending_p2p_works = {}
        for curr_dtype in sorted(input_split_sizes.keys()):
            send_buffer = send_buffers[curr_dtype]
            recv_buffer = recv_buffers[curr_dtype]
            handle = dist.all_to_all_single(
                recv_buffer,
                send_buffer,
                output_split_sizes[curr_dtype],
                input_split_sizes[curr_dtype],
                exec.process_group,
                async_op=True,
            )
            pending_p2p_works[curr_dtype] = handle
        exec.pending_comm_ops[instr.key] = (
            pending_p2p_works,
            recv_views,
            recv_buffers,
            recv_cpy_src_ptrs_tensors,
            recv_cpy_dst_ptrs_tensors,
            recv_cpy_src_offsets_tensors,
            recv_cpy_dst_offsets_tensors,
            recv_cpy_src_strides_tensors,
            recv_cpy_dst_strides_tensors,
            recv_cpy_dst_n_elems_tensors,
        )
    torch.cuda.nvtx.range_pop()


def _prepare_barrier(exec: AttentionExecutor, instr: BarrierInstr):
    exec.barrier_tensor = torch.tensor([1], device="cuda")


def _handle_barrier(exec: AttentionExecutor, instr: BarrierInstr):
    torch.cuda.nvtx.range_push("BarrierInstr")
    work = dist.all_reduce(
        exec.barrier_tensor, group=exec.process_group, async_op=True
    )
    work.wait()
    torch.cuda.nvtx.range_pop()


def _prepare_comm_wait(exec: AttentionExecutor, instr: CommWaitInstr):
    torch.cuda.nvtx.range_push("Prepare CommWaitInstr")
    if instr.stream is not None:
        if instr.stream not in exec.streams:
            exec.streams[instr.stream] = torch.cuda.Stream()
    torch.cuda.nvtx.range_pop()


def _handle_comm_wait(exec: AttentionExecutor, instr: CommWaitInstr):
    torch.cuda.nvtx.range_push("CommWaitInstr")
    (
        pending_p2p_works,
        recv_views,
        recv_buffers,
        recv_cpy_src_ptrs_tensors,
        recv_cpy_dst_ptrs_tensors,
        recv_cpy_src_offsets_tensors,
        recv_cpy_dst_offsets_tensors,
        recv_cpy_src_strides_tensors,
        recv_cpy_dst_strides_tensors,
        recv_cpy_dst_n_elems_tensors,
    ) = exec.pending_comm_ops[instr.key]

    if instr.stream is not None:
        stream = exec.streams[instr.stream]
    else:
        stream = None

    with torch.cuda.stream(stream):
        pending_p2p_works: Dict[int, dist.Work]
        for unique_dtype in sorted(pending_p2p_works.keys()):
            work = pending_p2p_works[unique_dtype]
            work.wait()
        # launch copy kernels
        for dtype, block_size in recv_cpy_src_ptrs_tensors.keys():
            if dtype == DType.FLOAT32:
                copy_fn = fused_copy_fp32_varlen
            elif dtype == DType.BFLOAT16:
                copy_fn = fused_copy_bf16_varlen
            elif dtype == DType.FLOAT16:
                copy_fn = fused_copy_fp16_varlen
            else:
                raise ValueError(f"Unsupported dtype {DType.as_str(dtype)}")
            copy_fn(
                recv_cpy_src_ptrs_tensors[(dtype, block_size)],
                recv_cpy_src_offsets_tensors[(dtype, block_size)],
                recv_cpy_src_strides_tensors[(dtype, block_size)],
                recv_cpy_dst_ptrs_tensors[(dtype, block_size)],
                recv_cpy_dst_offsets_tensors[(dtype, block_size)],
                recv_cpy_dst_strides_tensors[(dtype, block_size)],
                recv_cpy_dst_n_elems_tensors[(dtype, block_size)],
                block_size,
                128,
            )
        # handle pytorch fallbacks
        for dtype, recv_views in recv_views.items():
            for recv_view, target_view in recv_views:
                target_view.copy_(recv_view)
        exec.pending_comm_ops[instr.key] = {}
    torch.cuda.nvtx.range_pop()


def get_executor(
    n_heads_per_block: int,
    head_dim: int,
    exec_context: ExecutionContext,
    fw_exec_plan: Optional[ExecutionPlan] = None,
    fw_workload: Optional[WorkloadSpec] = None,
    bw_exec_plan: Optional[ExecutionPlan] = None,
    bw_workload: Optional[WorkloadSpec] = None,
    executor_impl="python",
    process_group: Optional[dist.ProcessGroup] = None,
    synchronous: bool = False,
    iteration_idx: Optional[int] = None,
    tp_rank: Optional[int] = None,
    use_cudagraph: bool = False,
):
    torch_rank = dist.get_rank(group=process_group)
    node_id = torch_rank // exec_context.n_devices_per_node
    local_device_id = torch_rank % exec_context.n_devices_per_node
    if executor_impl == "python":
        executor = AttentionExecutor(
            fw_exec_plan=fw_exec_plan,
            bw_exec_plan=bw_exec_plan,
            fw_workload=fw_workload,
            bw_workload=bw_workload,
            node_id=node_id,
            local_device_id=local_device_id,
            n_devices_per_node=exec_context.n_devices_per_node,
            n_nodes=exec_context.n_nodes,
            n_heads_per_block=n_heads_per_block,
            head_dim=head_dim,
            synchronous=synchronous,
            process_group=process_group,
            iteration_idx=iteration_idx,
            tp_rank=tp_rank,
            use_cudagraph=use_cudagraph,
        )
        # register handlers
        executor.register_handler(MemcpyInstr, _handle_memcpy)
        executor.register_handler(AttnInstr, _handle_attn)
        executor.register_handler(AttnBackwardInstr, _handle_attn_bw)
        executor.register_handler(AttnReductionInstr, _handle_reduction)
        executor.register_handler(SumInstr, _handle_sum)
        executor.register_handler(CommLaunchInstr, _handle_comm_launch)
        executor.register_handler(CommWaitInstr, _handle_comm_wait)
        executor.register_handler(BarrierInstr, _handle_barrier)
        executor.register_prep_handler(MemcpyInstr, _prepare_memcpy)
        executor.register_prep_handler(AttnInstr, _prepare_attn)
        executor.register_prep_handler(AttnBackwardInstr, _prepare_attn_bw)
        executor.register_prep_handler(AttnReductionInstr, _prepare_reduction)
        executor.register_prep_handler(SumInstr, _prepare_sum)
        executor.register_prep_handler(CommLaunchInstr, _prepare_comm_launch)
        executor.register_prep_handler(CommWaitInstr, _prepare_comm_wait)
        executor.register_prep_handler(BarrierInstr, _prepare_barrier)
        executor.check_all_handlers_registered()
    else:
        raise ValueError(f"Unsupported executor_impl: {executor_impl}")
    return executor
