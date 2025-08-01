# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
"""Pretrain GPT."""

import os
import torch
import gc
import warnings
from functools import partial
from contextlib import nullcontext
import inspect
from typing import Dict, Any, Optional

import torch.nn.functional as F
from torch.utils.data import DataLoader as PTDataLoader

from dcp.memory.host_caching_allocator import (
    apply_monkey_patch as patch_hca,
)
from dcp.ops.fused_ops import warmup_triton_ops
from dcp.runtime.flash_attention.utils import (
    lambda_mask_fn,
    shared_question_mask_fn,
    causal_blockwise_mask_fn,
    modality_specific_mask_fn,
    get_local_attn_ranges_zigzag,
    get_per_step_attn_range,
)

patch_hca()

# monkey patch before importing anything else
from monkey_patch import apply_monkey_patch

apply_monkey_patch()


def set_executor(self, *args, **kwargs):
    self.decoder.set_executor(*args, **kwargs)


from megatron.core.models.gpt import GPTModel
from megatron.core.packed_seq_params import PackedSeqParams

GPTModel.set_executor = set_executor


from typing import Union

from megatron.training import get_args
from megatron.training import print_rank_0
from megatron.training import get_timers
from megatron.training import get_tokenizer
from megatron.core import mpu
from megatron.core.enums import ModelType
from megatron.core.rerun_state_machine import get_rerun_state_machine
import megatron.legacy.model
from megatron.core.models.gpt import GPTModel
from megatron.training import pretrain
from megatron.core.utils import StragglerDetector
from megatron.core.transformer.spec_utils import import_module
from megatron.training.arguments import core_transformer_config_from_args
from megatron.training.yaml_arguments import core_transformer_config_from_yaml
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_decoder_block_spec,
    get_gpt_layer_local_spec,
    get_mlp_module_spec,
    ModuleSpec,
    TransformerLayer,
    TransformerLayerSubmodules,
    TENorm,
    is_te_min_version,
    FusedLayerNorm,
    SelfAttention,
    SelfAttentionSubmodules,
    TELayerNormColumnParallelLinear,
    TEDotProductAttention,
    TERowParallelLinear,
    IdentityOp,
    get_bias_dropout_add,
)

from dcp.data.hf_dataset import (
    HuggingFaceDataset,
    DummyMultiModelDataset,
    HuggingFaceDatasetConfig,
    OrderedBatchSampler,
)
from dcp.utils.common import trepr
from dcp.core.common import ExecutionContext, ModelSpec
from dcp.core.cost_model import (
    AttnRooflineCostModel,
    CommunicationCostModel,
)
from dcp.core.instructions import DType
from dcp.data.dataloader import DCPDataLoader, TrainingSpec

from dcp_args import add_dcp_extra_args


stimer = StragglerDetector()


def get_gpt_layer_with_transformer_engine_spec(
    num_experts: Optional[int] = None,
    moe_grouped_gemm: Optional[bool] = False,
    qk_layernorm: Optional[bool] = False,
    multi_latent_attention: Optional[bool] = False,
    fp8: Optional[str] = None,  # pylint: disable=unused-arguments
    moe_use_legacy_grouped_gemm: Optional[bool] = False,
    mask_type: Optional[str] = "causal",
) -> ModuleSpec:
    """Use this spec to use lower-level Transformer Engine modules (required for fp8 training).


    Args:
        num_experts (int, optional): Number of experts. Defaults to None.
        moe_grouped_gemm (bool, optional): To use Grouped GEMM. Defaults to False.
        qk_layernorm (bool, optional): To use layernorm for queries/keys. Defaults to False.
        fp8 (str, optional): Deprecated. For temporary Nemo compatibility.
        moe_use_legacy_grouped_gemm (bool, optional): Force use the legacy GroupedMLP.
                                                      Defaults to False.

    Returns:
        ModuleSpec: Module specification with TE modules
    """
    if fp8 is not None:
        warnings.warn(
            'The fp8 argument in "get_gpt_layer_with_transformer_engine_spec" has been deprecated'
            " and will be removed soon. Please update your code accordingly."
        )

    mlp = get_mlp_module_spec(
        use_te=True,
        num_experts=num_experts,
        moe_grouped_gemm=moe_grouped_gemm,
        moe_use_legacy_grouped_gemm=moe_use_legacy_grouped_gemm,
    )

    # TENorm significantly harms convergence when used
    # for QKLayerNorm if TE Version < 1.9;
    # we instead use the Apex implementation.
    qk_norm = TENorm if is_te_min_version("1.9.0") else FusedLayerNorm

    return ModuleSpec(
        module=TransformerLayer,
        submodules=TransformerLayerSubmodules(
            self_attention=ModuleSpec(
                module=SelfAttention,
                params={
                    "attn_mask_type": (
                        AttnMaskType.causal
                        if mask_type == "causal"
                        else AttnMaskType.custom_ranges
                    )
                },
                submodules=SelfAttentionSubmodules(
                    linear_qkv=TELayerNormColumnParallelLinear,
                    core_attention=TEDotProductAttention,
                    linear_proj=TERowParallelLinear,
                    q_layernorm=qk_norm if qk_layernorm else IdentityOp,
                    k_layernorm=qk_norm if qk_layernorm else IdentityOp,
                ),
            ),
            self_attn_bda=get_bias_dropout_add,
            pre_mlp_layernorm=TENorm if num_experts else IdentityOp,
            mlp=mlp,
            mlp_bda=get_bias_dropout_add,
        ),
    )


_MASK_FNS = {
    "causal": None,
    "lambda": lambda_mask_fn,
    "shared_question": shared_question_mask_fn,
    "causal_blockwise": causal_blockwise_mask_fn,
    "modality_specific": modality_specific_mask_fn,
    "modality_specific_sparse": partial(
        modality_specific_mask_fn, limit_image_attn_to_self=True
    ),
}


def _get_mask_fn(mask_type: str):
    if mask_type not in _MASK_FNS:
        raise ValueError(f"Unknown mask type: {mask_type}")
    return _MASK_FNS[mask_type]


def model_provider(
    pre_process=True, post_process=True
) -> Union[GPTModel, megatron.legacy.model.GPTModel]:
    """Builds the model.

    If you set the use_legacy_models to True, it will return the legacy GPT model and if not the mcore GPT model.

    Args:
        pre_process (bool, optional): Set to true if you need to compute embedings. Defaults to True.
        post_process (bool, optional): Set to true if you need to want to compute output logits/loss. Defaults to True.


    Returns:
        Union[GPTModel, megatron.legacy.model.GPTModel]: The returned model
    """
    args = get_args()
    use_te = args.transformer_impl == "transformer_engine"

    if args.record_memory_history:
        torch.cuda.memory._record_memory_history(
            True,
            # keep 100,000 alloc/free events from before the snapshot
            trace_alloc_max_entries=100000,
            # record stack information for the trace events
            trace_alloc_record_context=True,
        )

    print_rank_0("building GPT model ...")
    # Experimental loading arguments from yaml
    if args.yaml_cfg is not None:
        config = core_transformer_config_from_yaml(args, "language_model")
    else:
        config = core_transformer_config_from_args(args)

    if args.use_legacy_models:
        model = megatron.legacy.model.GPTModel(
            config,
            num_tokentypes=0,
            parallel_output=True,
            pre_process=pre_process,
            post_process=post_process,
        )
    else:  # using core models
        if args.spec is not None:
            transformer_layer_spec = import_module(args.spec)
        else:
            if args.num_experts:
                # Define the decoder block spec
                transformer_layer_spec = get_gpt_decoder_block_spec(
                    config, use_transformer_engine=use_te
                )
            else:
                # Define the decoder layer spec
                if use_te:
                    transformer_layer_spec = (
                        get_gpt_layer_with_transformer_engine_spec(
                            args.num_experts,
                            args.moe_grouped_gemm,
                            args.qk_layernorm,
                            args.multi_latent_attention,
                            args.fp8,
                            mask_type=args.dcp_mask_type,
                        )
                    )
                else:
                    transformer_layer_spec = get_gpt_layer_local_spec(
                        args.num_experts,
                        args.moe_grouped_gemm,
                        args.qk_layernorm,
                        args.multi_latent_attention,
                    )

        build_model_context = nullcontext
        build_model_context_args = {}
        if args.fp8_param_gather:
            try:
                from transformer_engine.pytorch import fp8_model_init

                build_model_context = fp8_model_init
                build_model_context_args["enabled"] = True

                # Check if fp8_model_init supports preserve_high_precision_init_val
                if (
                    "preserve_high_precision_init_val"
                    in inspect.signature(fp8_model_init).parameters
                ):
                    build_model_context_args[
                        "preserve_high_precision_init_val"
                    ] = True
            except:
                raise RuntimeError(
                    "--fp8-param-gather requires `fp8_model_init` from TransformerEngine, but not found."
                )

        with build_model_context(**build_model_context_args):
            model = GPTModel(
                config=config,
                transformer_layer_spec=transformer_layer_spec,
                vocab_size=args.padded_vocab_size,
                max_sequence_length=args.max_position_embeddings,
                pre_process=pre_process,
                post_process=post_process,
                fp16_lm_cross_entropy=args.fp16_lm_cross_entropy,
                parallel_output=True,
                share_embeddings_and_output_weights=not args.untie_embeddings_and_output_weights,
                position_embedding_type=args.position_embedding_type,
                rotary_percent=args.rotary_percent,
                rotary_base=args.rotary_base,
                rope_scaling=args.use_rope_scaling,
            )

    return model


def get_baseline_collate_fn(args):
    mask_fn = _get_mask_fn(args.dcp_mask_type)
    dp_rank = mpu.get_data_parallel_rank()
    dp_size = mpu.get_data_parallel_world_size()
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        ring_group = mpu.get_hierarchical_context_parallel_groups()[1]
        ring_rank = torch.distributed.get_rank(group=ring_group)
        ring_world_size = torch.distributed.get_world_size(group=ring_group)
    else:
        ring_rank = 0
        ring_world_size = 1

    def baseline_collate_fn(
        batch,
        input_key: str = "tokens",
        labels_key="labels",
        position_ids_key="position_ids",
        loss_mask_key="loss_mask",
        cu_seqlens_key="cu_seqlens",
        padded_cu_seqlens_key="padded_cu_seqlens",
        max_seqlen_key="max_seqlen",
        padded_max_seqlen_key="padded_max_seqlen",
        attn_ranges_key="attn_ranges",
        cu_seqlens_key_cpu="cu_seqlens_cpu",
    ):
        cp_size = mpu.get_context_parallel_world_size()
        input_tokens = []
        input_positions = []
        input_loss_masks = []
        label_tokens = []
        cu_seqlens = [0]
        padded_cu_seqlens = [0]
        max_seqlen = 0
        padded_max_seqlen = 0
        # first get local seqs
        raw_seqlens = []
        for seq_id, seq_dict in enumerate(batch):
            seqlen = len(seq_dict[input_key])
            raw_seqlens.append(seqlen)
        # greedy partitioning
        per_dp_rank_seqs = [[] for _ in range(dp_size)]
        per_dp_rank_tokens = [0 for _ in range(dp_size)]
        for seq_id, seqlen in enumerate(raw_seqlens):
            # select a rank with minimum tokens
            min_idx = per_dp_rank_tokens.index(min(per_dp_rank_tokens))
            per_dp_rank_seqs[min_idx].append(seq_id)
            per_dp_rank_tokens[min_idx] += seqlen
        local_seq_ids = per_dp_rank_seqs[dp_rank]

        raw_seqlens = []
        padded_seqlens = []
        mask_fn_infos = None
        for seq_id, seq_dict in enumerate(batch):
            if seq_id not in local_seq_ids:
                continue
            seq = seq_dict[input_key]
            raw_seqlens.append(len(seq))
            label = seq_dict[labels_key]
            position_ids = seq_dict[position_ids_key]
            loss_mask = seq_dict[loss_mask_key]
            cu_seqlens.append(len(seq) + cu_seqlens[-1])
            max_seqlen = max(max_seqlen, len(seq))
            # pad to 2 * cp_size
            pad_len = 2 * cp_size - len(seq) % (2 * cp_size)
            padded_seqlens.append(len(seq) + pad_len)
            seq = F.pad(seq, (0, pad_len), value=0)
            label = F.pad(label, (0, pad_len), value=0)
            position_ids = F.pad(position_ids, (0, pad_len), value=0)
            loss_mask = F.pad(loss_mask, (0, pad_len), value=0)
            input_tokens.append(seq)
            label_tokens.append(label)
            input_positions.append(position_ids)
            input_loss_masks.append(loss_mask)
            padded_cu_seqlens.append(len(seq) + padded_cu_seqlens[-1])
            padded_max_seqlen = max(padded_max_seqlen, len(seq))
            if args.dcp_mask_type.startswith("modality_specific"):
                assert "image_size" in seq_dict
                assert "text_lengths" in seq_dict
                fn_info_dict = {
                    "image_size": seq_dict["image_size"],
                    "text_lengths": seq_dict["text_lengths"],
                }
                if mask_fn_infos is None:
                    mask_fn_infos = []
                mask_fn_infos.append(fn_info_dict)
        # generate attn_ranges if necessary
        if mask_fn is not None:
            global_attn_ranges = mask_fn(
                raw_seqlens, padded_seqlens, torch.int32, mask_fn_infos
            )
            if global_attn_ranges.dim() == 3:
                global_attn_ranges = global_attn_ranges.permute(
                    1, 2, 0
                ).contiguous()
            else:
                global_attn_ranges = global_attn_ranges.transpose(
                    1, 0
                ).contiguous()
            local_attn_ranges = get_local_attn_ranges_zigzag(
                global_attn_ranges, padded_seqlens, ring_world_size, ring_rank
            ).cpu()
            padded_cu_seqlens_tensor_cpu = torch.tensor(
                padded_cu_seqlens, dtype=torch.int32
            ).cpu()
            per_step_attn_ranges = [
                get_per_step_attn_range(
                    i,
                    ring_rank,
                    ring_world_size,
                    local_attn_ranges,
                    padded_cu_seqlens_tensor_cpu,
                )
                for i in range(ring_world_size)
            ]
        else:
            per_step_attn_ranges = None
        return {
            input_key: torch.cat(input_tokens, dim=0).unsqueeze(0),
            labels_key: torch.cat(label_tokens, dim=0).unsqueeze(0),
            position_ids_key: torch.cat(input_positions, dim=0).unsqueeze(0),
            loss_mask_key: torch.cat(input_loss_masks, dim=0).unsqueeze(0),
            cu_seqlens_key: torch.tensor(cu_seqlens, dtype=torch.int32),
            padded_cu_seqlens_key: torch.tensor(
                padded_cu_seqlens, dtype=torch.int32
            ),
            max_seqlen_key: torch.tensor([max_seqlen], dtype=torch.int32),
            padded_max_seqlen_key: torch.tensor(
                [padded_max_seqlen], dtype=torch.int32
            ),
            attn_ranges_key: per_step_attn_ranges,
            cu_seqlens_key_cpu: torch.tensor(
                cu_seqlens, dtype=torch.int32, device="cpu"
            ),
        }

    return baseline_collate_fn


def get_batch_on_this_cp_rank(batch: Dict[str, Any]):
    cp_size = mpu.get_context_parallel_world_size()
    if cp_size > 1:
        cp_rank = mpu.get_context_parallel_rank()
        assert (
            "padded_cu_seqlens" in batch
        ), "padded_cu_seqlens must be in batch"
        padded_cu_seqlens = batch["padded_cu_seqlens"]
        for key, val in batch.items():
            if key in [
                "attention_mask",
                "cu_seqlens",
                "padded_cu_seqlens",
                "max_seqlen",
                "padded_max_seqlen",
                "attn_ranges",
            ]:
                continue
            local_vals = []
            for seq_id in range(len(padded_cu_seqlens) - 1):
                seq_vals = val[
                    :,
                    padded_cu_seqlens[seq_id] : padded_cu_seqlens[seq_id + 1],
                ]
                seq_vals = seq_vals.view(
                    val.shape[0],
                    2 * cp_size,
                    seq_vals.shape[1] // (2 * cp_size),
                    *seq_vals.shape[2:],
                )
                index = torch.tensor(
                    [cp_rank, (2 * cp_size - cp_rank - 1)],
                    device="cpu",
                    pin_memory=True,
                ).cuda(non_blocking=True)
                seq_vals = seq_vals.index_select(1, index)
                seq_vals = seq_vals.view(val.shape[0], -1, *seq_vals.shape[3:])
                local_vals.append(seq_vals)
            batch[key] = torch.cat(local_vals, dim=1)
    return batch


def get_batch(data_iterator):
    """Generate a batch."""

    args = get_args()

    # TODO: this is pretty hacky, find a better way
    if (not mpu.is_pipeline_first_stage()) and (
        not mpu.is_pipeline_last_stage()
    ):
        return None, None, None, None, None, None

    if args.enable_dcp:
        assert data_iterator is not None
        (
            data,
            fw_ep,
            bw_ep,
            fw_workload_spec,
            bw_workload_spec,
            training_spec,
        ) = next(data_iterator)
        attn_ranges = None
    else:
        data = next(data_iterator)
        assert "cu_seqlens" in data, "cu_seqlens must be in data"
        assert "padded_cu_seqlens" in data, "padded_cu_seqlens must be in data"
        cu_seqlens = data["cu_seqlens"].cuda(non_blocking=True)
        padded_cu_seqlens = data["padded_cu_seqlens"].cuda(non_blocking=True)
        max_seqlen = data["max_seqlen"].cuda(non_blocking=True)
        padded_max_seqlen = data["padded_max_seqlen"].cuda(non_blocking=True)
        if "attn_ranges" in data and data["attn_ranges"] is not None:
            attn_ranges = [
                attn_range.cuda(non_blocking=True)
                for attn_range in data["attn_ranges"]
            ]
        else:
            attn_ranges = None
    batch = {
        "tokens": data["tokens"].cuda(non_blocking=True),
        "labels": data["labels"].cuda(non_blocking=True),
        "loss_mask": data["loss_mask"].cuda(non_blocking=True),
        "attention_mask": (
            None
            if "attention_mask" not in data
            else data["attention_mask"].cuda(non_blocking=True)
        ),
        "position_ids": data["position_ids"].cuda(non_blocking=True),
        "attn_ranges": attn_ranges,
    }

    if args.enable_dcp:
        return (
            batch,
            fw_ep,
            bw_ep,
            fw_workload_spec,
            bw_workload_spec,
            training_spec,
        )
    else:
        batch["cu_seqlens"] = cu_seqlens
        batch["padded_cu_seqlens"] = padded_cu_seqlens
        batch["max_seqlen"] = max_seqlen
        batch["padded_max_seqlen"] = padded_max_seqlen
        # slice batch along sequence dimension for context parallelism
        data = get_batch_on_this_cp_rank(batch)
        return batch


# define spiky loss as a variation of 20% or more
SPIKY_LOSS_PERC = 0.2


def loss_func(loss_mask: torch.Tensor, output_tensor: torch.Tensor):
    """Loss function.

    Args:
        loss_mask (torch.Tensor): Used to mask out some portions of the loss
        output_tensor (torch.Tensor): The tensor with the losses

    Returns:
        the loss scalar for this micro-batch
        the number of non-padded tokens in this microbatch
        a dict containing reporting metrics on the loss and number of tokens across
            the data parallel ranks
    """
    args = get_args()

    losses = output_tensor.float()
    loss_mask = loss_mask.view(-1).float()
    total_tokens = loss_mask.sum()
    loss = torch.cat(
        [torch.sum(losses.view(-1) * loss_mask).view(1), total_tokens.view(1)]
    )

    if args.context_parallel_size > 1:
        torch.distributed.all_reduce(
            loss, group=mpu.get_context_parallel_group()
        )

    # Check individual rank losses are not NaN prior to DP all-reduce.
    rerun_state_machine = get_rerun_state_machine()
    if args.check_for_nan_in_loss_and_grad:
        rerun_state_machine.validate_result(
            result=loss[0],
            rejection_func=torch.isnan,
            message="found NaN in local forward loss calculation",
            tolerance=0.0,  # forward pass calculations are determinisic
            fatal=True,
        )
    # Check for spiky loss
    if args.check_for_spiky_loss:
        rerun_state_machine.validate_result(
            result=loss[0],
            rejection_func=partial(
                rerun_state_machine.is_spiky_loss, threshold=SPIKY_LOSS_PERC
            ),
            message="Spiky loss",
            tolerance=0.0,  # forward pass calculations are determinisic
            fatal=False,
        )
    # Reduce loss for logging.
    reporting_loss = loss.clone().detach()
    torch.distributed.all_reduce(
        reporting_loss, group=mpu.get_data_parallel_group()
    )

    local_num_tokens = loss[1].clone().detach().to(torch.int)
    return (
        loss[0] * args.context_parallel_size,
        local_num_tokens,
        {"lm loss": (reporting_loss[0], reporting_loss[1])},
    )


# _record_memory_called = False

# import pickle

# def oom_observer(device, alloc, device_alloc, device_free):
#     # snapshot right after an OOM happened
#     print('saving allocated state during OOM')
#     snapshot = torch.cuda.memory._snapshot()
#     pickle.dump(snapshot, open(f'/home/ubuntu/oom_snapshot_r{torch.distributed.get_rank()}.pickle', 'wb'))

# torch._C._cuda_attach_out_of_memory_observer(oom_observer)


def forward_step(data_iterator, model: GPTModel):
    """Forward training step.

    Args:
        data_iterator : Input data iterator
        model (GPTModel): The GPT Model
    """
    args = get_args()
    timers = get_timers()

    # global _record_memory_called
    # if not _record_memory_called:
    #     torch.cuda.memory._record_memory_history(
    #         True,
    #         # keep 100,000 alloc/free events from before the snapshot
    #         trace_alloc_max_entries=100000,
    #         # record stack information for the trace events
    #         trace_alloc_record_context=True,
    #     )

    # Get the batch.
    timers("batch-generator", log_level=2).start()
    global stimer
    with stimer(bdata=True):
        if args.enable_dcp:
            (
                data,
                fw_ep,
                bw_ep,
                fw_workload_spec,
                bw_workload_spec,
                training_spec,
            ) = get_batch(data_iterator)
            packed_seq_params = None
        else:
            data = get_batch(data_iterator)
            packed_seq_params = PackedSeqParams(
                qkv_format="thd",
                cu_seqlens_q=data[
                    "padded_cu_seqlens"
                ],  # thd + padding does not work
                cu_seqlens_kv=data[
                    "padded_cu_seqlens"
                ],  # thd + padding does not work
                # cu_seqlens_q_padded=data["padded_cu_seqlens"],
                # cu_seqlens_kv_padded=data["padded_cu_seqlens"],
                max_seqlen_q=data["padded_max_seqlen"],
                max_seqlen_kv=data["padded_max_seqlen"],
            )
            # attach attn_range to packed_seq_params
            if "attn_ranges" in data:
                packed_seq_params.attn_ranges = data["attn_ranges"]
            else:
                packed_seq_params.attn_ranges = None
            # attach padded_cu_seqlens_cpu to packed_seq_params
            packed_seq_params.cu_seqlens_cpu = (
                data["cu_seqlens_cpu"] if "cu_seqlens_cpu" in data else None
            )
        tokens = data["tokens"]
        position_ids = data["position_ids"]
        if "attention_mask" in data:
            attention_mask = data["attention_mask"]
        else:
            attention_mask = None
        labels = data["labels"]
        loss_mask = data["loss_mask"]
    timers("batch-generator").stop()

    if args.enable_dcp:
        torch.cuda.nvtx.range_push("SetExecutor")
        model.module.module.set_executor(
            fw_ep,
            bw_ep,
            fw_workload_spec,
            bw_workload_spec,
            training_spec,
            mpu.get_data_parallel_group(),
            iteration_idx=args.curr_iteration,
            tp_rank=mpu.get_tensor_model_parallel_rank(),
            use_cudagraph=args.dcp_use_cudagraph,
        )
        torch.cuda.nvtx.range_pop()

    with stimer:
        output_tensor = model(
            tokens,
            position_ids,
            attention_mask,
            labels=labels,
            packed_seq_params=packed_seq_params,
        )

    return output_tensor, partial(loss_func, loss_mask)


def is_dataset_built_on_rank():
    return (
        mpu.is_pipeline_first_stage() or mpu.is_pipeline_last_stage()
    ) and mpu.get_tensor_model_parallel_rank() == 0


def core_hf_dataset_config_from_args(args):
    tokenizer = get_tokenizer()

    assert len(args.data_path) == 1

    return HuggingFaceDatasetConfig(
        random_seed=args.seed,
        sequence_length=args.seq_length,
        data_path=args.data_path,
        split=args.split,
        path_to_cache=args.data_cache_path,
        tokenizer=tokenizer,
        num_preprocessing_workers=args.dcp_prefetch_planner_num_workers,
    )


def train_valid_test_datasets_provider(train_val_test_num_samples):
    """Build the train test and validation datasets.

    Args:
        train_val_test_num_samples : A list containing the number of samples in train test and validation.
    """
    args = get_args()

    config = core_hf_dataset_config_from_args(args)

    print_rank_0("> building train, validation, and test datasets for GPT ...")

    hf_dataset = config.data_path[0]

    if hf_dataset == "jchenyu/V2PE-Data-Long-MR-128K-Stats":
        train_ds = DummyMultiModelDataset(
            hf_dataset,
            num_samples=None,
            index_split=args.dcp_dataset_split,
            config=config,
            max_seq_length=args.seq_length,
            group=torch.distributed.group.WORLD,
        )
    else:
        train_ds = HuggingFaceDataset(
            hf_dataset,
            dataset_name=config.data_path[0],
            num_samples=None,
            index_split=args.dcp_dataset_split,
            text_key=args.dcp_dataset_text_key,
            config=config,
            max_seq_length=args.seq_length,
            group=torch.distributed.group.WORLD,
        )

    print_rank_0("> finished creating GPT datasets ...")

    n_devices_per_node = int(os.environ.get("LOCAL_WORLD_SIZE"))
    n_nodes = torch.distributed.get_world_size() // n_devices_per_node
    node_rank = torch.distributed.get_rank() // n_devices_per_node
    comm_cost_model = CommunicationCostModel(
        args.dcp_inter_node_lat,
        args.dcp_inter_node_bw,
        args.dcp_intra_node_lat,
        args.dcp_intra_node_bw,
    )
    comp_cost_model = AttnRooflineCostModel(
        args.dcp_max_tflops, args.dcp_max_mem_bw
    )
    buffer_size = args.dcp_prefetch_planner_num_workers
    listener_workers = args.dcp_prefetch_listener_num_workers
    dcp_size = args.data_parallel_size
    tp_size = args.tensor_model_parallel_size
    if args.enable_dcp:
        assert (
            args.context_parallel_size == 1
        ), "Context parallelism dimension should be merged with data parallelism dimensions"

    # calculate dcp node local rank
    dcp_group = mpu.get_data_parallel_group()
    dcp_ranks = torch.distributed.get_process_group_ranks(dcp_group)
    node_ids = [r // n_devices_per_node for r in dcp_ranks]
    local_dcp_ranks = sorted(
        [
            dcp_ranks[i]
            for i in range(len(node_ids))
            if node_ids[i] == node_rank
        ]
    )
    node_local_rank = local_dcp_ranks.index(torch.distributed.get_rank())

    exec_context = ExecutionContext(
        len(local_dcp_ranks), n_nodes, comm_cost_model, comp_cost_model
    )

    training_spec = TrainingSpec(
        exec_context,
        ModelSpec(
            args.kv_channels,
            args.num_attention_heads // tp_size,
            (
                args.num_query_groups
                if args.group_query_attention
                else args.num_attention_heads
            )
            // tp_size,
        ),
        dcp_size=dcp_size,
        mask_type=args.dcp_mask_type,
        mask_dtype=DType.INT,
        qkv_dtype=(
            DType.BFLOAT16 if args.bf16 else DType.FLOAT16
        ),  # Checkout the dtype
        block_size=args.dcp_block_size,
        head_block_size=args.dcp_head_block_size,
        tensor_parallel_size=args.tensor_model_parallel_size,
        pipeline_parallel_size=args.pipeline_model_parallel_size,
        use_block_size_heuristic=args.dcp_use_block_size_heuristic,
        mem_imbalance_epsilon=args.dcp_mem_imbalance_epsilon,
        comp_imbalance_epsilon=args.dcp_comp_imbalance_epsilon,
        inter_node_comp_imbalance_factor=args.dcp_inter_node_comp_imbalance_factor,
    )
    if args.enable_dcp:
        dataloader = DCPDataLoader(
            training_spec,
            train_ds,
            is_kv_host=torch.distributed.get_rank() == 0,
            node_rank=node_rank,
            node_local_rank=node_local_rank,
            node_size=n_nodes,
            dcp_rank=mpu.get_data_parallel_rank(),
            pp_rank=mpu.get_pipeline_model_parallel_rank(),
            tp_rank=mpu.get_tensor_model_parallel_rank(),
            start_poller=int(os.environ.get("LOCAL_RANK")) == 0,
            batch_sampler=OrderedBatchSampler(
                train_ds, args.dcp_global_batch_size_in_tokens
            ),
            num_workers=listener_workers,
            num_preprocess_workers=buffer_size,
            pin_memory=not args.dcp_use_cudagraph,  # pin memory interferes with cudagraph
            input_key="tokens",
            mask_fn=_get_mask_fn(args.dcp_mask_type),
            mask_fn_extra_keys=(
                ["image_size", "text_lengths"]
                if args.dcp_mask_type.startswith("modality_specific")
                else None
            ),
        )
    else:
        dataloader = PTDataLoader(
            train_ds,
            collate_fn=get_baseline_collate_fn(args),
            batch_sampler=OrderedBatchSampler(
                train_ds, args.dcp_global_batch_size_in_tokens
            ),
            num_workers=listener_workers,
            pin_memory=True,
        )
        dataloader = iter(dataloader)

    # before returning, warmup triton jit here
    if args.enable_dcp:
        warmup_triton_ops(
            args.dcp_head_block_size,
            1 if args.group_query_attention else args.dcp_head_block_size,
            args.kv_channels,
            torch.bfloat16 if args.bf16 else torch.float16,
        )
        gc.collect()
        torch.cuda.empty_cache()

    return dataloader, None, None


if __name__ == "__main__":

    # we set the distributed flag to True to also init data loader on
    # all TP ranks, to avoid broadcasting the data
    train_valid_test_datasets_provider.is_distributed = True

    pretrain(
        train_valid_test_datasets_provider,
        model_provider,
        ModelType.encoder_or_decoder,
        forward_step,
        args_defaults={"tokenizer_type": "GPT2BPETokenizer"},
        extra_args_provider=add_dcp_extra_args,
    )
