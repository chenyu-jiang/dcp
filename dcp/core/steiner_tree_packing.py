from typing import List, Tuple, Optional
import time
import logging

from ortools.sat.python import cp_model


def _boolean_and(model: cp_model.CpModel, a, b):
    c = model.NewBoolVar("{}_and_{}".format(a, b))
    model.add_min_equality(c, [a, b])
    return c


def pack_multicast_trees(
    n_devices: int,
    terminal_sets: List[Tuple[int, List[int]]],
    costs: List[int],
    n_stages: int,
    logger: Optional[logging.Logger] = None,
) -> List[List[int]]:
    n_stages = n_stages + 1
    model = cp_model.CpModel()
    # for each terminal set, create a list of variables representing the
    # edges in the multicast tree
    edge_vars = []
    for terminal_set_id in range(len(terminal_sets)):
        edges = []
        for i in range(n_devices):
            edges_i = []
            for j in range(n_devices):
                edges_j = []
                if i != j:
                    for stage_id in range(n_stages):
                        var = model.new_bool_var(
                            f"D{terminal_set_id}_N{i},{j}_S{stage_id}"
                        )
                        edges_j.append(var)
                edges_i.append(edges_j)
            edges.append(edges_i)
        edge_vars.append(edges)
    # require that connected edges is a tree
    for terminal_set_id in range(len(terminal_sets)):
        all_sum = 0
        for i in range(n_devices):
            for j in range(n_devices):
                if i != j:
                    all_sum += sum(edge_vars[terminal_set_id][i][j])
        model.add(all_sum == len(terminal_sets[terminal_set_id][1]))
    # create variables for connectedness of the multicast tree
    connectedness_vars = []
    for terminal_set_id in range(len(terminal_sets)):
        vars = [
            [
                model.new_bool_var(f"D{terminal_set_id}_C{i}_S{s}")
                for s in range(n_stages)
            ]
            for i in range(n_devices)
        ]
        connectedness_vars.append(vars)
    # create constraints on the connectedness vars
    for terminal_set_id in range(len(terminal_sets)):
        for i in range(n_devices):
            for s in range(1, n_stages):
                # any of the two conditions
                # 1. node is connected in the previous stage
                # 2. an incoming active edge is connected to a connected node
                #    in the previous stage
                model.add_max_equality(
                    connectedness_vars[terminal_set_id][i][s],
                    [
                        _boolean_and(
                            model,
                            edge_vars[terminal_set_id][j][i][s],
                            connectedness_vars[terminal_set_id][j][s - 1],
                        )
                        for j in range(n_devices)
                        if j != i
                    ]
                    + [connectedness_vars[terminal_set_id][i][s - 1]],
                )
            # in the first stage, only the src node is connected
            for i in range(n_devices):
                if i == terminal_sets[terminal_set_id][0]:
                    model.add(
                        connectedness_vars[terminal_set_id][i][0] == True
                    )
                else:
                    model.add(
                        connectedness_vars[terminal_set_id][i][0] == False
                    )
            # in the last stage, all terminal nodes should be connected
            for terminal_node in terminal_sets[terminal_set_id][1]:
                model.add(
                    connectedness_vars[terminal_set_id][terminal_node][-1]
                    == True
                )
    # create costs of each stage
    final_costs = []
    for stage_id in range(n_stages):
        send_costs = []
        recv_costs = []
        for d in range(n_devices):
            send_cost = 0
            recv_cost = 0
            for terminal_set_id in range(len(terminal_sets)):
                for i in range(n_devices):
                    if i != d:
                        send_cost += (
                            edge_vars[terminal_set_id][d][i][stage_id]
                            * costs[terminal_set_id]
                        )
                        recv_cost += (
                            edge_vars[terminal_set_id][i][d][stage_id]
                            * costs[terminal_set_id]
                        )
            send_cost_var = model.new_int_var(
                0, 1000000000, f"send_cost_{d}_S{stage_id}"
            )
            recv_cost_var = model.new_int_var(
                0, 1000000000, f"recv_cost_{d}_S{stage_id}"
            )
            model.add(send_cost_var == send_cost)
            model.add(recv_cost_var == recv_cost)
            send_costs.append(send_cost_var)
            recv_costs.append(recv_cost_var)
        max_send_cost = model.new_int_var(
            0, 1000000000, f"max_send_cost_S{stage_id}"
        )
        max_recv_cost = model.new_int_var(
            0, 1000000000, f"max_recv_cost_S{stage_id}"
        )
        model.add_max_equality(max_send_cost, send_costs)
        model.add_max_equality(max_recv_cost, recv_costs)
        max_cost = model.new_int_var(0, 1000000000, f"max_cost_S{stage_id}")
        model.add_max_equality(max_cost, [max_send_cost, max_recv_cost])
        final_costs.append(max_cost)
    final_cost = model.new_int_var(0, 1000000000, "final_cost")
    model.add_max_equality(final_cost, final_costs)
    model.minimize(final_cost)
    solver = cp_model.CpSolver()
    t = time.time()
    status = solver.Solve(model)
    if logger is not None:
        logger.debug(
            f"OR-tools solver finished in: {time.time() - t} seconds."
        )
    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        # get the multicast trees
        trees = [
            [[] for _ in range(n_stages - 1)]
            for _ in range(len(terminal_sets))
        ]
        for terminal_set_id in range(len(terminal_sets)):
            for i in range(n_devices):
                for j in range(n_devices):
                    if i != j:
                        for stage_id in range(n_stages):
                            if solver.Value(
                                edge_vars[terminal_set_id][i][j][stage_id]
                            ):
                                assert stage_id > 0
                                trees[terminal_set_id][stage_id - 1].append(
                                    (i, j)
                                )
        return trees
    else:
        return []


# def evaluate_schedule_quality(
#     n_devices: int,
#     trees: List[List[int]],
# ):
