from typing import Optional
import enum

import torch

from megatron.core.models.common.embeddings.rotary_pos_embedding import (
    apply_rotary_pos_emb,
)
import megatron.core.transformer.enums as tf_enum
from megatron.core.transformer.attention import Attention
from megatron.core.transformer.transformer_block import TransformerBlock
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.utils import is_te_min_version, get_te_version
from megatron.core.extensions.transformer_engine import TEDotProductAttention

from megatron.core.transformer import TransformerLayer
import megatron.training.training as mlm_training
from megatron.training.utils import print_rank_last

from dcp.runtime.flash_attention.executor import (
    DCPAttention,
    AttentionExecutor,
)
from dcp.data.dataloader import TrainingSpec
from dcp.core.instructions import ExecutionPlan
from dcp.core.block_table import WorkloadSpec
from dcp.runtime.flash_attention import get_executor


class AttnMaskType(enum.Enum):
    """Attention Mask Type"""

    padding = 1
    causal = 2
    no_mask = 3  # only used for TE
    padding_causal = 4  # only used for thd attention
    arbitrary = 5
    custom_ranges = 6


def Attention_set_executor(
    self,
    executor: AttentionExecutor,
):
    self.attention_executor = executor


def Attention_forward(
    self,
    hidden_states,
    attention_mask,
    key_value_states=None,
    inference_params=None,
    rotary_pos_emb=None,
    rotary_pos_cos=None,
    rotary_pos_sin=None,
    attention_bias=None,
    packed_seq_params=None,
    sequence_len_offset=None,
):
    """
    Perform a forward pass through the attention module.
    """

    # print("[RANK {}] hidden_states shape: {}".format(
    #     torch.distributed.get_rank(),
    #     hidden_states.shape)
    # )
    # print("[RANK {}] cu_seqlens: {}".format(
    #     torch.distributed.get_rank(),
    #     packed_seq_params.cu_seqlens_q)
    # )

    # hidden_states: [sq, b, h]
    if self.config.flash_decode:
        rotary_pos_emb = None
    else:
        assert rotary_pos_cos is None and rotary_pos_sin is None

    # For self attention we just duplicate the rotary_pos_emb if it isn't already
    if rotary_pos_emb is not None and not isinstance(rotary_pos_emb, tuple):
        rotary_pos_emb = (rotary_pos_emb,) * 2

    # =====================
    # Query, Key, and Value
    # =====================
    # Get the query, key and value tensors based on the type of attention -
    # self or cross attn.
    query, key, value = self.get_query_key_value_tensors(
        hidden_states, key_value_states
    )

    # ===================================================
    # Adjust key, value, and rotary_pos_emb for inference
    # ===================================================

    # This branch only runs in the decode phase of flash decoding and returns after the linear
    # projection. This conditional is not used in the prefill phase or non-flash-decoding cases.
    if (
        self.config.flash_decode
        and inference_params is not None
        and self.layer_number
        in inference_params.key_value_memory_dict  # Decode phase if key already exists
    ):
        assert inference_params.sequence_len_offset is not None
        inference_key_memory, inference_value_memory = (
            inference_params.key_value_memory_dict[self.layer_number]
        )
        output = self.flash_decoding(
            sequence_len_offset=inference_params.sequence_len_offset,
            query_layer=query,
            key_layer=key,
            value_layer=value,
            inference_key_memory=inference_key_memory,
            inference_value_memory=inference_value_memory,
            rotary_cos=rotary_pos_cos,
            rotary_sin=rotary_pos_sin,
        )
        out = output.transpose(0, 1).contiguous()
        context_layer = out.view(out.size(0), out.size(1), -1)
        output, bias = self.linear_proj(context_layer)
        return output, bias

    query, key, value, rotary_pos_emb, attn_mask_type = (
        self._adjust_key_value_for_inference(
            inference_params,
            query,
            key,
            value,
            rotary_pos_emb,
            rotary_pos_cos,
            rotary_pos_sin,
        )
    )

    if packed_seq_params is not None:
        query = query.squeeze(1)
        key = key.squeeze(1)
        value = value.squeeze(1)

    # ================================================
    # relative positional embedding (rotary embedding)
    # ================================================
    if rotary_pos_emb is not None and not self.config.flash_decode:
        q_pos_emb, k_pos_emb = rotary_pos_emb

        if packed_seq_params is not None:
            if packed_seq_params.cu_seqlens_q_padded is not None:
                cu_seqlens_q = packed_seq_params.cu_seqlens_q_padded
            else:
                cu_seqlens_q = packed_seq_params.cu_seqlens_q
            if packed_seq_params.cu_seqlens_kv_padded is not None:
                cu_seqlens_kv = packed_seq_params.cu_seqlens_kv_padded
            else:
                cu_seqlens_kv = packed_seq_params.cu_seqlens_kv
        else:
            cu_seqlens_q = cu_seqlens_kv = None
        query = apply_rotary_pos_emb(
            query, q_pos_emb, config=self.config, cu_seqlens=cu_seqlens_q
        )
        key = apply_rotary_pos_emb(
            key, k_pos_emb, config=self.config, cu_seqlens=cu_seqlens_kv
        )

        # TODO, can apply positional embedding to value_layer so it has
        # absolute positional embedding.
        # otherwise, only relative positional embedding takes effect
        # value_layer = apply_rotary_pos_emb(value_layer, k_pos_emb)

    # ==================================
    # core attention computation
    # ==================================

    if (
        not hasattr(self, "attention_executor")
        or self.attention_executor is None
    ):
        if self.checkpoint_core_attention and self.training:
            core_attn_out = self._checkpointed_attention_forward(
                query,
                key,
                value,
                attention_mask,
                attn_mask_type=attn_mask_type,
                attention_bias=attention_bias,
                packed_seq_params=packed_seq_params,
            )
        else:
            core_attn_out = self.core_attention(
                query,
                key,
                value,
                attention_mask,
                attn_mask_type=attn_mask_type,
                attention_bias=attention_bias,
                packed_seq_params=packed_seq_params,
            )
    else:
        # print("[RANK {}] Query memory: {}, Key memory: {} MB, Value memory: {} MB".format(
        #     torch.distributed.get_rank(),
        #     query.numel() * query.element_size() / 1e6,
        #     key.numel() * key.element_size() / 1e6,
        #     value.numel() * value.element_size() / 1e6
        # ))
        # print("[RANK {}] Query shape: {}, Key shape: {}, Value shape: {}".format(
        #     torch.distributed.get_rank(), query.shape, key.shape, value.shape
        # ))
        # QKV shape: [seqlen, bs, num heads, head dim]
        # [tokens, 1, num heads, head dim]
        # -> [tokens, num_head_blocks, n_heads_per_block, head dim]
        # -> [num_head_blocks, tokens, n_heads_per_block, head dim]
        # -> [num_head_blocks x tokens, n_heads_per_block, head dim]
        q = (
            query.squeeze(1)
            .reshape(
                query.shape[0],
                -1,
                self.attention_executor.n_heads_per_block,
                query.shape[-1],
            )
            .permute(1, 0, 2, 3)
            .reshape(
                -1, self.attention_executor.n_heads_per_block, query.shape[-1]
            )
            .contiguous()
        )
        is_gqa = (
            self.config.num_query_groups is not None
            and self.config.num_query_groups != self.config.num_attention_heads
        )
        n_heads_per_block_kv = (
            1 if is_gqa else self.attention_executor.n_heads_per_block
        )
        # [tokens, 1, 2, num heads kv, head dim]
        # -> [tokens, 2, num heads kv, head dim]
        # -> [tokens, 2, num_head_blocks, n_heads_per_block_kv, head dim]
        # -> [num_head_blocks, tokens, 2, n_heads_per_block_kv, head dim]
        # -> [num_head_blocks x tokens, 2, n_heads_per_block_kv, head dim]
        kv = (
            torch.stack((key.squeeze(1), value.squeeze(1)), dim=1)
            .reshape(
                key.shape[0],
                2,
                -1,
                n_heads_per_block_kv,
                key.shape[-1],
            )
            .permute(2, 0, 1, 3, 4)
            .reshape(-1, 2, n_heads_per_block_kv, key.shape[-1])
            .contiguous()
        )
        capture_fwd_graph = self.layer_number == 2
        capture_bwd_graph = self.layer_number == self.config.num_layers - 1
        core_attn_out = DCPAttention.apply(
            self.attention_executor,
            capture_fwd_graph,
            capture_bwd_graph,
            q,
            kv,
        )
        # out: [num_head_blocks x tokens, n_heads_per_block, head dim]
        # -> [num_head_blocks, tokens, n_heads_per_block, head dim]
        # -> [tokens, num_head_blocks, n_heads_per_block, head dim]
        # -> [tokens, 1, num heads x head dim]
        core_attn_out = (
            core_attn_out.reshape(
                -1,
                query.shape[0],
                self.attention_executor.n_heads_per_block,
                query.shape[-1],
            )
            .permute(1, 0, 2, 3)
            .reshape(query.shape[0], 1, -1)
        )

    if packed_seq_params is not None and packed_seq_params.qkv_format == "thd":
        # reshape to same output shape as unpacked case
        # (t, np, hn) -> (t, b=1, h=np*hn)
        # t is the pack size = sum (sq_i)
        # note that batch is a dummy dimension in the packed case
        core_attn_out = core_attn_out.reshape(core_attn_out.size(0), 1, -1)

    # =================
    # Output. [sq, b, h]
    # =================

    output, bias = self.linear_proj(core_attn_out)

    return output, bias


_executor = None


def TransformerBlock_set_executor(
    self,
    fw_exec_plan: ExecutionPlan,
    bw_exec_plan: ExecutionPlan,
    fw_workload: WorkloadSpec,
    bw_workload: WorkloadSpec,
    training_spec: TrainingSpec,
    dcp_group: Optional[torch.distributed.ProcessGroup] = None,
    executor_impl: str = "python",
    synchronous: bool = False,
    iteration_idx: Optional[int] = None,
    tp_rank: Optional[int] = None,
    use_cudagraph: bool = False,
):
    executor = get_executor(
        training_spec.head_block_size,
        training_spec.model_spec.head_dim,
        training_spec.exec_context,
        fw_exec_plan,
        fw_workload,
        bw_exec_plan,
        bw_workload,
        executor_impl=executor_impl,
        process_group=dcp_group,
        synchronous=synchronous,
        iteration_idx=iteration_idx,
        tp_rank=tp_rank,
        use_cudagraph=use_cudagraph,
    )
    executor.prepare(forward=True)
    executor.log_memory_usage()
    self.executor = executor
    for layer in self.layers:
        if hasattr(layer, "set_executor"):
            layer.set_executor(executor)
    global _executor
    _executor = executor


def TransformerLayer_set_executor(self, *args, **kwargs):
    self.self_attention.set_executor(*args, **kwargs)


def Training_training_log_wrapper(
    loss_dict,
    total_loss_dict,
    learning_rate,
    decoupled_learning_rate,
    iteration,
    loss_scale,
    report_memory_flag,
    skipped_iter,
    grad_norm,
    params_norm,
    num_zeros_in_grad,
):
    if _executor is not None:
        _executor.deallocate_all_buffers()
    mlm_training.training_log_orig(
        loss_dict,
        total_loss_dict,
        learning_rate,
        decoupled_learning_rate,
        iteration,
        loss_scale,
        report_memory_flag,
        skipped_iter,
        grad_norm,
        params_norm,
        num_zeros_in_grad,
    )
    print_rank_last("Training loss: {}".format(loss_dict["lm loss"]))
    # print_rank_last("Training loss total: {}".format(total_loss_dict["lm loss"]))
    # dev_name = torch.cuda._get_nvml_device_index(torch.cuda.current_device())
    # snapshot = torch.cuda.memory._snapshot(device=torch.cuda.current_device())
    # from pickle import dumps

    # if iteration % 10 == 0:
    #     with open(f"snapshot_iter{iteration}_dev{dev_name}.pkl", "wb") as f:
    #         f.write(dumps(snapshot))
    return True


def TEDotProductAttention_forward(
    self,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor,
    attn_mask_type: AttnMaskType,
    attention_bias: torch.Tensor = None,
    packed_seq_params: PackedSeqParams = None,
):
    """Forward."""
    packed_seq_kwargs = (
        {
            key: getattr(packed_seq_params, key)
            for key in self.kept_packed_seq_params
        }
        if packed_seq_params is not None
        else {}
    )
    if hasattr(packed_seq_params, "attn_ranges"):
        packed_seq_kwargs["attn_ranges_per_step"] = (
            packed_seq_params.attn_ranges
        )
    if hasattr(packed_seq_params, "cu_seqlens_cpu"):
        packed_seq_kwargs["cu_seqlens_q_cpu"] = (
            packed_seq_params.cu_seqlens_cpu
        )
        packed_seq_kwargs["cu_seqlens_kv_cpu"] = (
            packed_seq_params.cu_seqlens_cpu
        )
    # overwrite self.qkv_format depending on self.config.apply_rope_fusion, which can be set
    # after init
    if self.config.apply_rope_fusion and is_te_min_version(
        "0.13.0", check_equality=False
    ):
        self.qkv_format = "bshd"

    qkv_format = packed_seq_kwargs.get("qkv_format", self.qkv_format)

    # WAR for peak memory usage.
    # See https://gitlab-master.nvidia.com/ADLR/megatron-lm/-/merge_requests/2388
    if self.config.apply_rope_fusion and qkv_format == "bshd":
        query, key, value = [
            x.transpose(0, 1).contiguous() for x in (query, key, value)
        ]
        # In PyTorch, the following two tensors are in fact the same:
        #   Tensor with shape (1, S, H, D) and stride (S*H*D, H*D, D, 1)
        #   Tensor with shape (1, S, H, D) and stride (H*D, H*D, D, 1)
        # Stride for a dimension that is 1 has no meaning, so tensors created two different ways
        # can have same shape but different strides.
        # We unify them to the first one to pass the stride check in TE
        if (
            value.shape == key.shape
            and value.shape[0] == 1
            and value.stride() != key.stride()
        ):
            value = value.as_strided(value.shape, key.stride())

    attention_bias_kwargs = {}
    if attention_bias is not None:
        assert is_te_min_version("1.2.0"), (
            f"Transformer-Engine v{get_te_version()} must be >= 1.2.0 to support"
            "`attention_bias`."
        )
        attention_bias_kwargs = dict(
            core_attention_bias_type="post_scale_bias",
            core_attention_bias=attention_bias,
        )

    if self.te_forward_mask_type:
        if qkv_format == "thd" and is_te_min_version("1.7.0"):
            # thd format uses flash attention with cuDNN kernel which requires is_padding=True,
            # so the only acceptable mask types are `padding_causal` and `padding`. These do not
            # necessarily indicate there are padded tokens in the sequence.
            if attn_mask_type == AttnMaskType.causal:
                attn_mask_type = AttnMaskType.padding_causal
            elif attn_mask_type == AttnMaskType.no_mask:
                attn_mask_type = AttnMaskType.padding
        core_attn_out = super(TEDotProductAttention, self).forward(
            query,
            key,
            value,
            attention_mask,
            attn_mask_type=attn_mask_type.name,
            **attention_bias_kwargs,
            **packed_seq_kwargs,
        )
    else:
        core_attn_out = super(TEDotProductAttention, self).forward(
            query,
            key,
            value,
            attention_mask,
            **attention_bias_kwargs,
            **packed_seq_kwargs,
        )

    if self.config.apply_rope_fusion and qkv_format == "bshd":
        return core_attn_out.transpose(0, 1)
    else:
        return core_attn_out


def apply_monkey_patch():
    tf_enum.AttnMaskType = AttnMaskType
    Attention.set_executor = Attention_set_executor
    Attention.forward = Attention_forward
    TransformerLayer.set_executor = TransformerLayer_set_executor

    TransformerBlock.set_executor = TransformerBlock_set_executor

    TEDotProductAttention.forward = TEDotProductAttention_forward

    mlm_training.training_log_orig = mlm_training.training_log
    mlm_training.training_log = Training_training_log_wrapper
