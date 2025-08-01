# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy
import torch
import torch.distributed as dist
import datasets
from datasets import load_dataset

# from megatron.core.datasets.utils import Split
from dcp.utils.logger import read_env_bool

logger = logging.getLogger(__name__)

_PAD_TOKEN_ID = -1


@dataclass
class HuggingFaceDatasetConfig:
    random_seed: int
    sequence_length: int
    data_path: str
    tokenizer: Any
    mmap_bin_files: Optional[bool] = False
    split: Optional[Any] = None
    split_matrix: Optional[List[Tuple[float, float]]] = field(
        init=False, default=None
    )
    train_dataset_path: Optional[str] = ""
    valid_dataset_path: Optional[str] = ""
    test_dataset_path: Optional[str] = ""
    path_to_cache: Optional[str] = None
    add_extra_token_to_sequence: Optional[bool] = False
    create_attention_mask: Optional[bool] = False
    num_preprocessing_workers: Optional[int] = 1


def _get_masks_and_position_ids(
    data: torch.Tensor,
    create_attention_mask: bool,
):
    """Build masks and position id for left to right model.

    Args:
        data (torch.Tensor): The data tenor that holds the tokens from the dataset

        eod_token (int): ID of the token to that is considered the EOD

        reset_position_ids (bool): Switch to reset the document position ID's

        reset_attention_mask (bool): Switch to reset the attention mask

        eod_mask_loss (bool): Switch to enable the EOD mask loss

        create_attention_mask (bool): Switch to enable the attention masks generation. Can be
            disabled if attention kernel generates masks by itself.

    Returns:
        torch.Tensor: Attention mask needed to be used for Attention

        torch.Tensor: The mask used for loss value during training

        torch.Tensor: The position ID's of the token
    """
    seq_length = data.numel()

    if create_attention_mask:
        attention_mask = torch.tril(
            torch.ones((seq_length, seq_length), device=data.device)
        ).unsqueeze(0)
    else:
        attention_mask = None

    # Loss mask.
    loss_mask = torch.ones(seq_length, dtype=torch.float, device=data.device)

    # Position ids.
    position_ids = torch.arange(
        seq_length, dtype=torch.long, device=data.device
    )

    if attention_mask is not None:
        # Convert attention mask to binary:
        attention_mask = attention_mask < 0.5

    return attention_mask, loss_mask, position_ids


class HuggingFaceDataset(torch.utils.data.Dataset):
    """The base GPT dataset

    Args:
        indexed_dataset (IndexedDataset): The huggingface dataset aruond which to build the HuggingFaceDataset

        indexed_indices (numpy.ndarray): The set of the documents indices to expose

        num_samples (Optional[int]): The number of samples to draw from the indexed dataset. When None, build as many samples as correspond to one epoch.

        index_split (Split): The indexed_indices Split

        config (GPTDatasetConfig): The config
    """

    def __init__(
        self,
        dataset: Any,
        dataset_name: Optional[str],
        num_samples: Optional[int],
        index_split: Any,
        text_key: str,
        config: Optional[HuggingFaceDatasetConfig] = None,
        max_seq_length: int = 4096,
        group: Optional[dist.ProcessGroup] = None,
        needs_tokenization: bool = True,
    ) -> None:
        if group is None:
            group = dist.group.WORLD
        if isinstance(dataset, str):
            if dist.is_initialized():
                init_with_global_rank = read_env_bool(
                    "DCP_DATASET_INIT_WITH_GLOBAL_RANK"
                )
                local_rank = (
                    os.environ.get("LOCAL_RANK", 0)
                    if not init_with_global_rank
                    else dist.get_rank(group=group)
                )
                if local_rank == 0:
                    # let local rank 0 download the dataset
                    dataset = load_dataset(dataset, split=index_split)
                    dist.barrier(
                        group=group, device_ids=[torch.cuda.current_device()]
                    )
                else:
                    dist.barrier(
                        group=group, device_ids=[torch.cuda.current_device()]
                    )
                    time.sleep(1)
                    dataset = load_dataset(dataset, split=index_split)
            else:
                dataset = load_dataset(dataset, split=index_split)
        self.dataset = dataset
        self.dataset_name = dataset_name
        self.num_samples = num_samples
        self.index_split = index_split
        self.text_key = text_key
        self.tokens_key = f"{self.text_key}_tokens"
        self.config = config
        self.max_seq_length = max_seq_length

        self.ordered = True

        try:
            self._pad_token_id = self.config.tokenizer.pad
        except Exception:
            self._pad_token_id = _PAD_TOKEN_ID

        if needs_tokenization:
            self.dataset = self._tokenize(group)

    def __len__(self) -> int:
        """Abstract method implementation

        Returns:
            int: The length of the dataset
        """
        return len(self.dataset)

    def _tokenize(self, group):
        # pre-tokenize the dataset
        init_with_global_rank = read_env_bool(
            "DCP_DATASET_INIT_WITH_GLOBAL_RANK"
        )
        if dist.is_initialized():
            local_rank = (
                int(os.environ.get("LOCAL_RANK", 0))
                if not init_with_global_rank
                else dist.get_rank(group=group)
            )
        else:
            local_rank = 0
        hf_home = os.environ.get(
            "HF_HOME", os.path.expanduser("~/.cache/huggingface")
        )
        cache_dir = os.path.join(
            hf_home, "tokenized_datasets", self.dataset_name
        )
        if not os.path.exists(cache_dir) and local_rank == 0:
            os.makedirs(cache_dir)
        if dist.is_initialized():
            dist.barrier(group=group, device_ids=[torch.cuda.current_device()])

        def tokenize(sample):
            sample_text = sample[self.text_key]
            if isinstance(sample_text, list):
                # role and content
                text = ""
                for conversation in sample_text:
                    # we just randomly concat then since this will not
                    # greatly affect seqlen
                    text += 'role: "' + conversation["role"] + '", '
                    text += 'content: "' + conversation["content"] + '".'
                text = self.config.tokenizer.tokenize(text)
                sample[self.tokens_key] = text
            else:
                sample[self.tokens_key] = self.config.tokenizer.tokenize(
                    sample[self.text_key]
                )
            return sample

        def filter_long_sample(sample):
            return len(sample[self.tokens_key]) <= self.max_seq_length

        cache_file_name = os.path.join(
            cache_dir,
            f"tokenized_sp{self.index_split}_n{self.num_samples}_msl{self.max_seq_length}.arrow",
        )
        if local_rank == 0:
            if os.path.exists(cache_file_name):
                tokenized_dataset = datasets.load_from_disk(cache_file_name)
            else:
                tokenized_dataset = self.dataset.map(
                    tokenize,
                    num_proc=self.config.num_preprocessing_workers,
                )
                tokenized_dataset = tokenized_dataset.filter(
                    filter_long_sample,
                    num_proc=self.config.num_preprocessing_workers,
                )
                tokenized_dataset.save_to_disk(cache_file_name)
                time.sleep(1)
            if dist.is_initialized():
                dist.barrier(
                    group=group, device_ids=[torch.cuda.current_device()]
                )
        else:
            if dist.is_initialized():
                dist.barrier(
                    group=group, device_ids=[torch.cuda.current_device()]
                )
            tokenized_dataset = datasets.load_from_disk(cache_file_name)
        return tokenized_dataset

    def __getitem__(self, idx: Optional[int]) -> Dict[str, torch.Tensor]:
        """Abstract method implementation

        Args:
            idx (Optioal[int]): The index into the dataset

        Returns:
            Dict[str, torch.Tensor]: The sample information wrapped in a dictionary
        """
        if self.tokens_key not in self.dataset[idx]:
            raise ValueError(
                "Tokens key {} not found in the dataset. Existing keys: {}".format(
                    self.tokens_key, self.dataset[idx].keys()
                )
            )
        if idx is None:
            # Batch padding sequence so the index does not matter
            # text, _ = self._query_document_sample_shuffle_indices(0)
            text = self.dataset[0][self.tokens_key]
        else:
            # text, _ = self._query_document_sample_shuffle_indices(idx)
            text = self.dataset[idx][self.tokens_key]

        text = torch.tensor(text, dtype=torch.long)

        if self.config.add_extra_token_to_sequence:
            tokens = text[:-1].contiguous()
            labels = text[1:].contiguous()
        else:
            tokens = text
            labels = torch.roll(text, shifts=-1, dims=0)
            labels[-1] = self._pad_token_id

        attention_mask, loss_mask, position_ids = _get_masks_and_position_ids(
            tokens,
            self.config.create_attention_mask,
        )

        # For padded sequences, mask the loss
        loss_mask[labels == self._pad_token_id] = 0.0

        # For padded sequences, ensure the embedding layer can map the token ID
        tokens[tokens == self._pad_token_id] = 0
        labels[labels == self._pad_token_id] = 0

        # Batch padding sequence so we mask the loss
        if idx is None:
            loss_mask = torch.zeros_like(loss_mask)

        if self.config.create_attention_mask:
            return {
                "tokens": tokens,
                "labels": labels,
                "attention_mask": attention_mask,
                "loss_mask": loss_mask,
                "position_ids": position_ids,
            }
        else:
            return {
                "tokens": tokens,
                "labels": labels,
                "loss_mask": loss_mask,
                "position_ids": position_ids,
            }

    def get_seq_len(self, idx: int) -> int:
        seqlen = len(self.dataset[idx][self.tokens_key])
        return seqlen

    def _get_num_epochs(self, num_tokens_per_epoch: int) -> int:
        """Calculate the number of epochs

        Args:
            num_tokens_per_epoch (int): The number of tokens in a single epoch

        Returns:
            int: The number of epochs
        """
        num_epochs = 1
        num_tokens = num_tokens_per_epoch
        if self.num_samples is None:
            return num_epochs
        else:
            num_tokens_requested = (
                self.num_samples * self.config.sequence_length
            ) + self.config.add_extra_token_to_sequence
            while num_tokens < num_tokens_requested:
                num_epochs += 1
                num_tokens += num_tokens_per_epoch
        return num_epochs


class DummyMultiModelDataset(HuggingFaceDataset):
    def __init__(
        self,
        dataset: Any,
        num_samples: Optional[int],
        index_split: Any,
        config: Optional[HuggingFaceDatasetConfig] = None,
        max_seq_length: int = 4096,
        group: Optional[dist.ProcessGroup] = None,
    ):
        super().__init__(
            dataset,
            dataset,
            num_samples,
            index_split,
            "",
            config,
            max_seq_length,
            group,
            needs_tokenization=False,
        )

    def __getitem__(self, idx: Optional[int]) -> Dict[str, torch.Tensor]:
        """Abstract method implementation

        Args:
            idx (Optioal[int]): The index into the dataset

        Returns:
            Dict[str, torch.Tensor]: The sample information wrapped in a dictionary
        """
        assert idx is not None
        sample = self.dataset[idx]

        img_size = sample["image_size"]
        text_lengths = sample["text_lengths"]

        # construct sample by concatenating text and image tokens
        tokens = []
        for i, text_length in enumerate(text_lengths):
            if text_length > 0:
                text = torch.randint(0, 50000, (text_length,)).tolist()
                tokens.extend(text)
            if i < len(text_lengths) - 1:
                tokens.extend(torch.randint(0, 50000, (img_size,)).tolist())

        text = torch.tensor(tokens, dtype=torch.long)

        if self.config.add_extra_token_to_sequence:
            tokens = text[:-1].contiguous()
            labels = text[1:].contiguous()
        else:
            tokens = text
            labels = torch.roll(text, shifts=-1, dims=0)
            labels[-1] = self._pad_token_id

        attention_mask, loss_mask, position_ids = _get_masks_and_position_ids(
            tokens,
            self.config.create_attention_mask,
        )

        # For padded sequences, mask the loss
        loss_mask[labels == self._pad_token_id] = 0.0

        # For padded sequences, ensure the embedding layer can map the token ID
        tokens[tokens == self._pad_token_id] = 0
        labels[labels == self._pad_token_id] = 0

        # Batch padding sequence so we mask the loss
        if idx is None:
            loss_mask = torch.zeros_like(loss_mask)

        if self.config.create_attention_mask:
            return {
                "tokens": tokens,
                "labels": labels,
                "attention_mask": attention_mask,
                "loss_mask": loss_mask,
                "position_ids": position_ids,
                "image_size": img_size,
                "text_lengths": text_lengths,
            }
        else:
            return {
                "tokens": tokens,
                "labels": labels,
                "loss_mask": loss_mask,
                "position_ids": position_ids,
                "image_size": img_size,
                "text_lengths": text_lengths,
            }

    def get_seq_len(self, idx: int) -> int:
        sample = self.dataset[idx]
        total_text_length = sample["total_text_length"]
        total_img_length = sample["total_img_length"]
        return total_text_length + total_img_length


class OrderedBatchSampler:
    def __init__(
        self,
        dataset: HuggingFaceDataset,
        batch_size_in_tokens: int,
        shuffle: bool = True,
        seed: int = 24,
    ):
        self.batch_size = batch_size_in_tokens
        self.dataset = dataset
        self.shuffle = shuffle
        self.seed = seed
        self.consumed_samples = 0
        if self.shuffle:
            self.rng = numpy.random.default_rng(seed=self.seed)
            self.data_order = self.rng.permutation(len(self.dataset))
        else:
            self.data_order = numpy.arange(len(self.dataset))

    def __iter__(self):
        self.consumed_samples = 0
        return self

    def __next__(self):
        while True:
            if self.consumed_samples >= len(self.dataset):
                raise StopIteration
            data_indices = []
            total_tokens = 0
            while True:
                if self.consumed_samples >= len(self.dataset):
                    break
                next_data_idx = int(self.data_order[self.consumed_samples])
                seq_len = self.dataset.get_seq_len(next_data_idx)
                if total_tokens + seq_len > self.batch_size:
                    break
                data_indices.append(next_data_idx)
                total_tokens += seq_len
                self.consumed_samples += 1
            if total_tokens < self.batch_size * 0.75:
                # too small, skip
                continue
            if len(data_indices) == 0:
                raise StopIteration
            return data_indices


class MockGPTLowLevelDataset:

    seed: int = 0
    size: int = 100000
    max_sequence_length: int = 4096

    def __init__(self, tokenizer) -> None:
        self.tokenizer = tokenizer
        rng = numpy.random.default_rng(seed=self.seed)
        self.sequence_lengths = rng.integers(
            low=1,
            high=self.max_sequence_length,
            size=self.size,
            dtype=numpy.int32,
        )

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int) -> numpy.number:
        length = self.sequence_lengths[idx]
        sample = numpy.int64(
            numpy.concatenate(
                [numpy.arange(length - 1) + 1, [self.tokenizer.eod]]
            )
        )
        return sample

    def get(
        self, idx: int, offset: int = 0, length: Optional[int] = None
    ) -> numpy.ndarray:
        if length is None:
            length = self.sequence_lengths[idx] - offset
        return self[idx][offset : offset + length]
