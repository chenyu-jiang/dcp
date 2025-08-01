# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Type

from dcp.core.instructions import *  # noqa: F403
from dcp.utils.logger import create_logger, read_env_bool

DEBUG_LOG_EXECUTOR = read_env_bool("DCP_DEBUG_LOG_EXECUTOR", default=False)

_LOCAL_BUFFER_TYPES_FW = [
    BufferType.LOCAL_Q,
    BufferType.LOCAL_KV,
]
_LOCAL_BUFFER_TYPES_BW = [
    BufferType.LOCAL_Q,
    BufferType.LOCAL_KV,
    BufferType.LOCAL_OUT,
    BufferType.LOCAL_LSE,
    BufferType.LOCAL_dOUT,
]
_TMP_BUFFER_TYPES = [
    BufferType.BUFFER_Q,
    BufferType.BUFFER_KV,
    BufferType.BUFFER_OUT,
    BufferType.BUFFER_LSE,
    BufferType.BUFFER_dOUT,
    BufferType.BUFFER_dQ,
    BufferType.BUFFER_dKV,
]


class DCPExecutor:
    """
    Executes the distributed block computation according to instructions.
    """

    def __init__(
        self,
        node_id: int = None,
        local_device_id: int = None,
        n_devices_per_node: int = None,
        n_nodes: int = None,
        synchronous: bool = False,
        iteration_idx: Optional[int] = None,
        tp_rank: Optional[int] = None,
    ):
        # rank is optional, only used for debugging
        self.buffer_slots = []
        self._instruction_handlers = {}
        self.node_id = node_id
        self.local_device_id = local_device_id
        self.n_devices_per_node = n_devices_per_node
        self.n_nodes = n_nodes
        # if synchronous, cuda device synchronization is performed after
        # each instruction, mainly used for debugging
        self.synchronous = synchronous
        # register default handlers
        tp_rank_str = f"T{tp_rank}" if tp_rank is not None else ""
        if iteration_idx is not None:
            log_file = f"executor/N{node_id}D{local_device_id}{tp_rank_str}_I{iteration_idx}.log"
        else:
            log_file = (
                f"executor/N{node_id}D{local_device_id}{tp_rank_str}.log"
            )
        if DEBUG_LOG_EXECUTOR:
            self.logger = create_logger(
                "DCPExecutor",
                prefix=f"N{node_id}D{local_device_id}",
                log_file=log_file,
            )
        else:
            self.logger = None
        self.instr_index = 0
        self.return_values: Dict[int, Any] = {}
        self.buffers: Dict[str, torch.Tensor] = {}
        self.buffer_n_blocks: Dict[str, int] = {}
        self.buffer_block_strides: Dict[str, int] = {}
        self.buffer_block_sizes: Dict[str, int] = {}
        self.buffer_per_token_shape: Dict[str, Tuple[int, ...]] = {}
        self.tp_rank = tp_rank
        self.iteration_idx = iteration_idx

    @property
    def rank(self):
        return self.get_rank(self.node_id, self.local_device_id)

    def get_rank(self, node_id: int, local_device_id: int):
        return node_id * self.n_devices_per_node + local_device_id

    def reverse_get_rank(self, rank: int):
        return rank // self.n_devices_per_node, rank % self.n_devices_per_node

    def init_forward_input(self, local_q, local_kv):
        self.buffers[BufferType.LOCAL_Q].copy_(local_q)
        self.buffers[BufferType.LOCAL_KV].copy_(local_kv)

    def init_backward_input(self, local_dout):
        self.buffers[BufferType.LOCAL_dOUT].copy_(local_dout)

    def deallocate_buffers(self):
        for buffer_type in self.buffers.keys():
            if buffer_type in _TMP_BUFFER_TYPES:
                self.buffers[buffer_type] = None

    def init_buffers(self, execution_plan: ExecutionPlan):
        for buffer_type, buffer_info in execution_plan.buffer_info.items():
            self.buffer_n_blocks[buffer_type] = buffer_info.n_blocks
            self.buffer_block_strides[buffer_type] = buffer_info.block_numel
            self.buffer_block_sizes[buffer_type] = buffer_info.block_size
            self.buffer_per_token_shape[buffer_type] = (
                buffer_info.per_token_shape
            )
            if (
                buffer_type not in self.buffers
                or self.buffers[buffer_type] is None
            ):
                if self.logger:
                    self.logger.debug(
                        f"Creating buffer {buffer_type} with shape "
                        f"{buffer_info.buffer_shape} and dtype {buffer_info.dtype}, "
                        f"per_token_shape: {buffer_info.per_token_shape}"
                    )
                self.buffers[buffer_type] = self._create_buffer(
                    buffer_info.buffer_shape, buffer_info.dtype
                )

    def execute(self, execution_plan: ExecutionPlan, is_forward: bool = True):
        self.execution_plan = execution_plan
        # initialize buffers
        self.init_buffers(execution_plan)
        for instr_index, instruction in enumerate(execution_plan.instructions):
            if self.logger:
                self.logger.debug("Executing instruction: %s", instruction)
            self.instr_index = instr_index
            self._execute_instruction(instruction)
            if self.synchronous and self.logger:
                self.logger.debug("Instruction complete.")
        if self.logger:
            self.logger.debug("Execution complete.")
        return self.return_values

    def register_handler(
        self, instruction_type: Type[BlockInstrBase] | Any, handler
    ):
        if not issubclass(instruction_type, BlockInstrBase):
            raise TypeError(
                f"Instruction type must be a subclass of PipeInstruction, "
                f"got {instruction_type.__name__}"
            )
        if instruction_type in self._instruction_handlers:
            raise ValueError(
                f"Instruction handler for {instruction_type.__name__} "
                "already registered."
            )
        self._instruction_handlers[instruction_type] = handler

    @classmethod
    def _get_leaf_subclasses(cls, instruction_type: Type[BlockInstrBase]):
        subclasses = instruction_type.__subclasses__()
        if subclasses:
            for subclass in subclasses:
                yield from cls._get_leaf_subclasses(subclass)
        else:
            yield instruction_type

    @classmethod
    def get_all_needed_handlers(cls):
        needed_handlers = set(cls._get_leaf_subclasses(BlockInstrBase))
        return [x.__name__ for x in needed_handlers]

    def check_all_handlers_registered(self):
        for instruction_type in self._get_leaf_subclasses(BlockInstrBase):
            if instruction_type not in self._instruction_handlers:
                raise ValueError(
                    "No handler registered for instruction "
                    f"{instruction_type.__name__}"
                )

    def register_synchronization_handler(self, handler):
        self.synchronize = handler

    def synchronize(self):
        raise NotImplementedError("Synchronization handler not registered")

    def _create_buffer(self, buffer_shape: Tuple[int, ...], buffer_dtype: int):
        raise NotImplementedError("DCPExecutor._create_buffer not implemented")

    def _execute_instruction(self, instruction: BlockInstrBase):
        handler = self._instruction_handlers.get(type(instruction))
        if handler is None:
            raise ValueError(
                "No handler registered for instruction "
                f"{type(instruction).__name__}"
            )
        handler(self, instruction)
        if self.synchronous:
            self.synchronize()
