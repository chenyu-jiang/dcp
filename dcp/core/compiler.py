import time
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np

from dcp.core.block_table import BlockType, WorkloadSpec
from dcp.core.common import ExecutionContext
from dcp.core.instructions import ExecutionPlan
from dcp.core.scheduler import convert_forward_to_backward, schedule
from dcp.graph_partition.graph import (
    construct_and_partition_graph_multiconstraint,
    HGPSolver,
)
from dcp.utils.logger import create_logger, read_env_bool

COMPILER_LOG_WORKLOAD_SPEC = read_env_bool(
    "DCP_LOG_WORKLOAD_SPEC", default=False
)
COMPILER_LOG_INSTRS = read_env_bool("DCP_LOG_INSTRS", default=False)

DEBUG_SAVE_PARTITION_RESULTS = read_env_bool(
    "DCP_DEBUG_SAVE_PARTITION_RESULTS", default=False
)
DEBUG_LOAD_PARTITION_RESULTS = read_env_bool(
    "DCP_DEBUG_LOAD_PARTITION_RESULTS", default=False
)
DEBUG_LOAD_PARTITION_RESULTS_FILE = os.environ.get(
    "DCP_DEBUG_LOAD_PARTITION_RESULTS_FILE", None
)
assert not (DEBUG_SAVE_PARTITION_RESULTS and DEBUG_LOAD_PARTITION_RESULTS)
if DEBUG_LOAD_PARTITION_RESULTS_FILE is None:
    DEBUG_LOAD_PARTITION_RESULTS_FILE = str(
        Path.home() / ".dcp_partition_results"
    )
DEBUG_LOAD_EXEC_PLANS = read_env_bool(
    "DCP_DEBUG_LOAD_EXEC_PLANS", default=False
)
DEBUG_LOAD_EXEC_PLANS_FILE = os.environ.get(
    "DCP_DEBUG_LOAD_EXEC_PLANS_FILE", None
)
if DEBUG_LOAD_EXEC_PLANS and DEBUG_LOAD_EXEC_PLANS_FILE is None:
    raise ValueError(
        "DEBUG_LOAD_EXEC_PLANS is set but DEBUG_LOAD_EXEC_PLANS_FILE is not provided"
    )
DEBUG_VISUALIZE_WORKLOAD_SPEC = read_env_bool(
    "DCP_DEBUG_VISUALIZE_WORKLOAD_SPEC", default=False
)
DEBUG_VISUALIZE_WORKLOAD_SPEC_DIR = os.environ.get(
    "DCP_DEBUG_VISUALIZE_WORKLOAD_SPEC_DIR", None
)
if DEBUG_VISUALIZE_WORKLOAD_SPEC and DEBUG_VISUALIZE_WORKLOAD_SPEC_DIR is None:
    raise ValueError(
        "DCP_DEBUG_VISUALIZE_WORKLOAD_SPEC is set but DCP_DEBUG_VISUALIZE_WORKLOAD_SPEC_DIR is not provided"
    )
DEBUG_OVERRIDE_PARTITION_WITH_HEURISTIC = read_env_bool(
    "DCP_DEBUG_OVERRIDE_PARTITION_WITH_HEURISTIC", default=False
)


@dataclass
class CompilerConfig:
    mem_imbalance_epsilon: float = 0.2
    comp_imbalance_epsilon: float = 0.2
    inter_node_comp_imbalance_factor: float = 5.0
    reduction_schedule_algo: str = "delayed"
    solver: HGPSolver = HGPSolver.KAHYPAR


class InstrCompiler:
    def __init__(
        self,
        context: ExecutionContext,
        config: Optional[CompilerConfig] = None,
        iteration_idx: Optional[int] = None,
    ):
        self.context = context
        self.config = config or CompilerConfig()
        self.iteration_idx = None
        if iteration_idx is not None:
            self.iteration_idx = iteration_idx
            log_file = f"compiler/compiler_iter{iteration_idx}.log"
        else:
            log_file = f"compiler/compiler.log"
        self.logger = create_logger(name="InstrCompiler", log_file=log_file)
        self.logger.debug(f"Inited compiler with config: {self.config}")

    def compile(
        self, workload: WorkloadSpec, generate_backward: bool = True
    ) -> Tuple[
        WorkloadSpec,
        WorkloadSpec,
        Dict[Tuple[int, int], ExecutionPlan],
        Dict[Tuple[int, int], ExecutionPlan],
    ]:
        t_compile = time.time()
        if DEBUG_LOAD_EXEC_PLANS:
            import pickle

            with open(DEBUG_LOAD_EXEC_PLANS_FILE, "rb") as f:
                (
                    workload,
                    bw_workload,
                    fw_execution_plans,
                    bw_execution_plans,
                ) = pickle.load(f)
            return (
                workload,
                bw_workload,
                fw_execution_plans,
                bw_execution_plans,
            )
        # Step 1. calculate workload assignment
        if (
            not DEBUG_LOAD_PARTITION_RESULTS
            and not DEBUG_OVERRIDE_PARTITION_WITH_HEURISTIC
        ):
            self.logger.debug("Start graph partitioning...")
            start_time = time.time()
            (
                device_to_workload_map,
                device_to_input_map,
                device_to_output_map,
            ) = construct_and_partition_graph_multiconstraint(
                self.context,
                workload,
                mem_epsilon=self.config.mem_imbalance_epsilon,
                comp_epsilon=self.config.comp_imbalance_epsilon,
                inter_node_comp_epsilon_factor=self.config.inter_node_comp_imbalance_factor,
                solver=self.config.solver,
                logger=self.logger,
            )
            end_time = time.time()
            self.logger.debug(
                f"Graph partitioning took {end_time - start_time} seconds."
            )

        if DEBUG_OVERRIDE_PARTITION_WITH_HEURISTIC:
            from dcp.graph_partition.graph import assign_device_by_heuristic

            (
                device_to_workload_map,
                device_to_input_map,
                device_to_output_map,
            ) = assign_device_by_heuristic(
                self.context,
                workload,
            )

        if DEBUG_SAVE_PARTITION_RESULTS:
            import pickle

            with open(DEBUG_LOAD_PARTITION_RESULTS_FILE, "wb") as f:
                pickle.dump(
                    (
                        workload,
                        device_to_workload_map,
                        device_to_input_map,
                        device_to_output_map,
                    ),
                    f,
                )

        if DEBUG_LOAD_PARTITION_RESULTS:
            import pickle

            with open(DEBUG_LOAD_PARTITION_RESULTS_FILE, "rb") as f:
                (
                    workload,
                    device_to_workload_map,
                    device_to_input_map,
                    device_to_output_map,
                ) = pickle.load(f)

        # fill in workload assignment
        workload.work_to_device_map = {}
        for d, work_ids in device_to_workload_map.items():
            for w in work_ids:
                workload.work_to_device_map[w] = d

        # fill in initial buffer positions based on input/output assignment
        def _head_seq_block(data_id, is_input):
            if is_input:
                return (
                    workload.block_mapping.input_id_to_meta[data_id].head_id,
                    workload.block_mapping.input_id_to_meta[data_id].seq_id,
                    workload.block_mapping.input_id_to_meta[data_id].block_id,
                )
            else:
                return (
                    workload.block_mapping.output_id_to_meta[data_id].head_id,
                    workload.block_mapping.output_id_to_meta[data_id].seq_id,
                    workload.block_mapping.output_id_to_meta[data_id].block_id,
                )

        input_id_to_buffer_index = {}
        output_id_to_buffer_index = {}
        local_cu_seqlens_per_block = {}
        for d in device_to_input_map.keys():
            input_ids_assigned: List[int] = device_to_input_map[d]
            output_ids_assigned: List[int] = device_to_output_map[d]
            input_id_to_buffer_index[d] = {}
            output_id_to_buffer_index[d] = {}
            cu_seqlens = []
            # assume input user buffer is of shape [n_heads, tokens, head_dim]
            # where tokens are sorted by local seq_id, block_id
            q_buffer_input_ids = [
                i
                for i in input_ids_assigned
                if workload.block_mapping.input_id_to_meta[i].type
                == BlockType.Q
            ]
            q_buffer_input_ids.sort(key=lambda x: _head_seq_block(x, True))
            for local_input_index, input_id in enumerate(q_buffer_input_ids):
                input_id_to_buffer_index[d][input_id] = local_input_index
                cu_seqlens.append(
                    workload.block_mapping.input_id_to_meta[input_id].n_tokens
                )

            cu_seqlens = np.cumsum([0] + cu_seqlens).tolist()
            local_cu_seqlens_per_block[d] = cu_seqlens

            kv_buffer_input_ids = [
                i
                for i in input_ids_assigned
                if workload.block_mapping.input_id_to_meta[i].type
                == BlockType.KV
            ]
            kv_buffer_input_ids.sort(key=lambda x: _head_seq_block(x, True))
            for local_input_index, input_id in enumerate(kv_buffer_input_ids):
                input_id_to_buffer_index[d][input_id] = local_input_index

            out_buffer_output_ids = [
                i
                for i in output_ids_assigned
                if workload.block_mapping.output_id_to_meta[i].type
                == BlockType.Out
            ]
            out_buffer_output_ids.sort(key=lambda x: _head_seq_block(x, False))
            for local_output_index, output_id in enumerate(
                out_buffer_output_ids
            ):
                output_id_to_buffer_index[d][output_id] = local_output_index

            lse_buffer_output_ids = [
                i
                for i in output_ids_assigned
                if workload.block_mapping.output_id_to_meta[i].type
                == BlockType.LSE
            ]
            lse_buffer_output_ids.sort(key=lambda x: _head_seq_block(x, False))
            for local_output_index, output_id in enumerate(
                lse_buffer_output_ids
            ):
                output_id_to_buffer_index[d][output_id] = local_output_index

        workload.block_mapping.input_id_to_buffer_index = (
            input_id_to_buffer_index
        )
        workload.block_mapping.output_id_to_buffer_index = (
            output_id_to_buffer_index
        )
        all_devices = (
            set(device_to_workload_map.keys())
            .union(device_to_input_map.keys())
            .union(device_to_output_map.keys())
        )
        for d in all_devices:
            input_ids = device_to_input_map.get(d, [])
            output_ids = device_to_output_map.get(d, [])
            work_ids = device_to_workload_map.get(d, [])
            self.logger.debug(
                f"Device {d} assigned {len(input_ids)} input blocks, "
                f"{len(output_ids)} output blocks "
                f"(total mem: {sum(workload.output_sizes[i] for i in output_ids) + sum(workload.input_sizes[i] for i in input_ids)}), "
                f"and {len(work_ids)} compute blocks (total workload: {sum(workload.workloads[i] for i in work_ids)})."
            )

        if DEBUG_VISUALIZE_WORKLOAD_SPEC:
            from dcp.utils.visualization import (
                visualize_workload_spec_figure,
            )

            visualize_workload_spec_figure(
                DEBUG_VISUALIZE_WORKLOAD_SPEC_DIR,
                workload,
            )

        # Step 2. generate forward schedule
        start_time = time.time()
        fw_execution_plans, work_to_stage_map = schedule(
            self.context,
            workload,
            self.context.comm_cost_model,
            logger=self.logger,
        )
        end_time = time.time()
        self.logger.debug(
            f"Forward scheduling took {end_time - start_time} seconds."
        )
        workload.work_to_stage_map = work_to_stage_map

        # Step 3. generate backward schedule
        # for now, use the same workload cost as forward
        if generate_backward:
            start_time = time.time()
            bw_workload = convert_forward_to_backward(
                workload, workload.workloads, logger=self.logger
            )
            end_time = time.time()
            self.logger.debug(
                f"Backward workload conversion took {end_time - start_time} seconds."
            )
            start_time = time.time()
            bw_execution_plans, bw_work_to_stage_map = schedule(
                self.context,
                bw_workload,
                self.context.comm_cost_model,
                logger=self.logger,
                is_forward=False,
            )
            end_time = time.time()
            self.logger.debug(
                f"Backward scheduling took {end_time - start_time} seconds."
            )
            bw_workload.work_to_stage_map = bw_work_to_stage_map
        else:
            bw_workload = None
            bw_execution_plans = None
        if COMPILER_LOG_INSTRS:
            self.logger.debug("Forward Execution Plan:")
            for d, exec_plan in fw_execution_plans.items():
                self.logger.debug(
                    f"Device {d}: {len(exec_plan.instructions)} instructions"
                )
                for instr in exec_plan.instructions:
                    self.logger.debug(f"\t{instr}")
            if generate_backward:
                self.logger.debug("Backward Execution Plan:")
                for d, exec_plan in bw_execution_plans.items():
                    self.logger.debug(
                        f"Device {d}: {len(exec_plan.instructions)} instructions"
                    )
                    for instr in exec_plan.instructions:
                        self.logger.debug(f"\t{instr}")
        if COMPILER_LOG_WORKLOAD_SPEC:
            from dcp.utils.visualization import visualize_workload_spec_tty

            visualize_workload_spec_tty(self.logger, workload)
        # fill in local cu seqlens
        for d, exec_plan in fw_execution_plans.items():
            exec_plan: ExecutionPlan
            if d not in local_cu_seqlens_per_block:
                continue
            exec_plan.local_cu_seqlens = local_cu_seqlens_per_block[d]
        if generate_backward:
            for d, exec_plan in bw_execution_plans.items():
                if d not in local_cu_seqlens_per_block:
                    continue
                exec_plan.local_cu_seqlens = local_cu_seqlens_per_block[d]
        t_compile = time.time() - t_compile
        self.logger.debug(f"Compilation took {t_compile} seconds. ")
        return workload, bw_workload, fw_execution_plans, bw_execution_plans
