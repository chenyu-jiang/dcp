import os
from collections import defaultdict
from termcolor import colored

from dcp.core.block_table import WorkloadSpec
from dcp.core.block_table import BlockType
from dcp.core.scheduler import link_sequence_block_to_input_output_id
from dcp.utils.logger import Logger, logger


def visualize_workload_spec_tty(
    file_logger: Logger, workload_spec: WorkloadSpec
):
    """
    Visualizes the workload spec in the terminal.
    """
    all_devices = (
        set(workload_spec.input_to_device_map.values())
        .union(set(workload_spec.output_to_device_map.values()))
        .union(set(workload_spec.work_to_device_map.values()))
    )
    if not workload_spec.work_to_stage_map:
        n_stages = 1
        work_to_stage_map = {
            work_id: 0 for work_id in range(len(workload_spec.workloads))
        }
    else:
        n_stages = max(workload_spec.work_to_stage_map.values()) + 1
        work_to_stage_map = workload_spec.work_to_stage_map
    colors = [
        "red",
        "green",
        "yellow",
        "blue",
        "magenta",
        "cyan",
        "white",
        "dark_grey",
    ]
    node_attributes = [
        ["bold"],
        ["dark"],
    ]
    stage_attributes = [
        None,
        ["underline"],
        ["strike"],
    ]
    device_color_map = {}
    n_nodes = len(set(d[0] for d in all_devices))
    n_devices_per_node = len(all_devices) // n_nodes
    if n_devices_per_node > len(colors):
        logger.warning(
            f"Too many devices ({len(all_devices)}) to visualize in tty, "
            f"only visualizing the first {len(colors)} devices on each node"
        )
        colors = colors + ["white"] * (len(all_devices) - len(colors))
    assert n_stages <= len(
        stage_attributes
    ), f"Too many stages ({n_stages}) to visualize in tty."
    assert n_nodes <= len(
        node_attributes
    ), f"Too many nodes ({n_nodes}) to visualize in tty."
    for node_id in range(n_nodes):
        for device_id in range(n_devices_per_node):
            for stage_id in range(n_stages):
                color = colors[device_id]
                node_attr = node_attributes[node_id]
                stage_attr = stage_attributes[stage_id]
                if stage_attr is not None:
                    attribute = node_attr + stage_attr
                else:
                    attribute = node_attr
                device_color_map[((node_id, device_id), stage_id)] = (
                    color,
                    attribute,
                )

    s = "\x1b[0m\n"  # reset any previous color
    for device in sorted(all_devices):
        for stage_id in range(n_stages):
            color, attribute = device_color_map[(device, stage_id)]
            s += colored(
                f"Device {device} Stage {stage_id}"
                + (", " if stage_id < n_stages - 1 else ""),
                color,
                attrs=attribute,
            )
        s += "\n"
    if file_logger is not None:
        file_logger.info(s)
    # logger.info(s)

    rev_block_mapping = link_sequence_block_to_input_output_id(workload_spec)

    # print sequence by sequence
    s = "\x1b[0m\n"  # reset any previous color
    for seq_id in range(rev_block_mapping.n_total_sequences):
        s += colored(f"Seq {seq_id}:\n", "white", attrs=["bold"])
        align_digits = len(
            str(rev_block_mapping.n_blocks_per_sequence[seq_id] - 1)
        )

        def _pad(s):
            return str(s).zfill(align_digits)

        for q_block_id in range(
            rev_block_mapping.n_blocks_per_sequence[seq_id]
        ):
            for head_block_id in range(rev_block_mapping.n_head_blocks):
                q_input_id = rev_block_mapping.seq_head_block_id_to_q_input_id[
                    (seq_id, head_block_id, q_block_id)
                ]
                q_device = workload_spec.input_to_device_map[
                    rev_block_mapping.seq_head_block_id_to_q_input_id[
                        (seq_id, head_block_id, q_block_id)
                    ]
                ]
                if workload_spec.block_mapping.input_id_to_buffer_index:
                    q_buffer_idx = (
                        workload_spec.block_mapping.input_id_to_buffer_index[
                            q_device
                        ][q_input_id]
                    )
                    buffer_str = f" (BufferIdx: {q_buffer_idx})"
                else:
                    buffer_str = ""
                color, attribute = device_color_map[(q_device, 0)]
                s += colored(
                    f"\tQ{_pad(q_block_id)}H{_pad(head_block_id)}{buffer_str}: ",
                    color,
                    attrs=attribute,
                )
                workloads_for_current_q = [
                    (
                        work_id,
                        workload_spec.block_mapping.input_id_to_meta[
                            kv_id
                        ].block_id,
                    )
                    for work_id, _, q_id, kv_id, _, _ in rev_block_mapping.seq_id_to_work_ids[
                        seq_id
                    ]
                    if q_id == q_input_id
                ]
                workloads_for_current_q.sort(key=lambda x: x[1])
                for i, (work_id, kv_block_id) in enumerate(
                    workloads_for_current_q
                ):
                    work_device = workload_spec.work_to_device_map[work_id]
                    stage_id = work_to_stage_map[work_id]
                    color, attribute = device_color_map[
                        (work_device, stage_id)
                    ]
                    s += colored(
                        f"Q{_pad(q_block_id)}H{_pad(head_block_id)}KV{_pad(kv_block_id)}",
                        color,
                        attrs=attribute,
                    )
                    if i < len(workloads_for_current_q) - 1:
                        s += ", "
                out_device = workload_spec.output_to_device_map[
                    rev_block_mapping.seq_head_block_id_to_output_id[
                        (seq_id, head_block_id, q_block_id)
                    ]
                ]
                color, attribute = device_color_map[(out_device, 0)]
                out_id = rev_block_mapping.seq_head_block_id_to_output_id[
                    (seq_id, head_block_id, q_block_id)
                ]
                if workload_spec.block_mapping.output_id_to_buffer_index:
                    out_buffer_idx = (
                        workload_spec.block_mapping.output_id_to_buffer_index[
                            out_device
                        ][out_id]
                    )
                    buffer_str = f" (BufferIdx: {out_buffer_idx})"
                else:
                    buffer_str = ""
                s += colored(
                    f" -> OUT{_pad(q_block_id)}H{_pad(head_block_id)}{buffer_str}\n",
                    color,
                    attrs=attribute,
                )
        s += "\n"
    if file_logger is not None:
        file_logger.info(s)
    # logger.info(s)


def visualize_workload_spec_figure(
    out_dir: str,
    workload_spec: WorkloadSpec,
    cluster_weight: float = 2,
    color_by=["device", "seq", "head", "node_type"],
):
    """
    Converts the workload spec to a networkx graph and draw a figure.
    workload_spec is supposed to represent forward pass.
    """
    import networkx as nx

    G = nx.Graph()
    for work_id, _ in enumerate(workload_spec.workloads):
        work_device = workload_spec.work_to_device_map[work_id]
        work_meta = workload_spec.block_mapping.work_id_to_meta[work_id]
        work_name = f"S{work_meta.seq_id}H{work_meta.head_id}Q{work_meta.q_id}KV{work_meta.kv_id}"
        G.add_node(
            f"W{work_id}",
            device=work_device,
            seq_id=work_meta.seq_id,
            head_id=work_meta.head_id,
            node_type="W",
            label=work_name,
        )
    for input_id, input_device in workload_spec.input_to_device_map.items():
        input_meta = workload_spec.block_mapping.input_id_to_meta[input_id]
        if input_meta.type == BlockType.Q:
            input_name = f"S{input_meta.seq_id}H{work_meta.head_id}Q{input_meta.block_id}"
        else:
            input_name = f"S{input_meta.seq_id}H{work_meta.head_id}KV{input_meta.block_id}"
        G.add_node(
            f"I{input_id}",
            device=input_device,
            seq_id=input_meta.seq_id,
            head_id=input_meta.head_id,
            node_type="I",
            label=input_name,
        )
    for output_id, output_device in workload_spec.output_to_device_map.items():
        output_meta = workload_spec.block_mapping.output_id_to_meta[output_id]
        if output_meta.type == BlockType.LSE:
            # LSE is too small, ignore
            continue
        assert output_meta.type == BlockType.Out
        out_name = f"S{output_meta.seq_id}H{output_meta.head_id}OUT{output_meta.block_id}"
        G.add_node(
            f"O{output_id}",
            device=output_device,
            seq_id=output_meta.seq_id,
            head_id=output_meta.head_id,
            node_type="O",
            label=out_name,
        )
    # add edges between work, input and outputs
    for work_id, _ in enumerate(workload_spec.workloads):
        input_ids = workload_spec.work_unit_input_map[work_id]
        output_ids = workload_spec.work_unit_output_map[work_id]
        for input_id in input_ids:
            G.add_edge(
                f"I{input_id}",
                f"W{work_id}",
                weight=workload_spec.input_sizes[input_id],
                spring_weight=1,
            )
        for output_id in output_ids:
            out_meta = workload_spec.block_mapping.output_id_to_meta[output_id]
            if out_meta.type == BlockType.Out:
                G.add_edge(
                    f"W{work_id}",
                    f"O{output_id}",
                    weight=workload_spec.output_sizes[output_id],
                    spring_weight=1,
                )
    # create a copy of the graph to add cluster edges
    G_cluster = G.copy()
    per_device_nodes = defaultdict(list)
    for node in G.nodes:
        device = G.nodes[node]["device"]
        per_device_nodes[device].append(node)
    for device, nodes in per_device_nodes.items():
        # add a dummy node for the device
        # G_cluster.add_node(f"D{device}", label=f"Device {device}")
        # for node in nodes:
        #     G_cluster.add_edge(f"D{device}", node, spring_weight=cluster_weight)
        for node0 in nodes:
            for node1 in nodes:
                if node0 != node1:
                    G_cluster.add_edge(
                        node0, node1, spring_weight=cluster_weight * 4
                    )
    # also cluster by machine
    per_machine_nodes = defaultdict(list)
    for node in G.nodes:
        device = G.nodes[node]["device"]
        machine = device[0]
        per_machine_nodes[machine].append(node)
    for machine, nodes in per_machine_nodes.items():
        # add a dummy node for the machine
        G_cluster.add_node(f"N{machine}", label=f"Node {machine}")
        for node in nodes:
            G_cluster.add_edge(
                f"N{machine}", node, spring_weight=cluster_weight * 100
            )
        # for node0 in nodes:
        #     for node1 in nodes:
        #         if node0 != node1:
        #             G_cluster.add_edge(node0, node1, spring_weight=cluster_weight * 0.5)
    # run spring layout
    print("Running spring layout...")
    node_pos = nx.spring_layout(G_cluster, weight="spring_weight")
    print("Drawing the graph...")
    # draw the graph
    import matplotlib.pyplot as plt

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    colors = list(plt.cm.get_cmap("tab20", 20).colors)
    for color_by_key in color_by:
        # assign a color to each node
        node_color_map = {}
        # create a legend
        legend_handles = []
        legend_labels = []
        if color_by_key == "device":
            device_color_map = {}
            for i, device in enumerate(sorted(list(per_device_nodes.keys()))):
                device_color_map[device] = colors[i]
                legend_handles.append(
                    plt.Line2D(
                        [0],
                        [0],
                        marker="o",
                        color="w",
                        markerfacecolor=colors[i],
                        markersize=10,
                    )
                )
                legend_labels.append(f"Device {device}")
            for node in G.nodes:
                device = G.nodes[node]["device"]
                node_color_map[node] = device_color_map[device]
        elif color_by_key == "seq":
            unique_seq_ids = set(G.nodes[node]["seq_id"] for node in G.nodes)
            for i, seq_id in enumerate(sorted(list(unique_seq_ids))):
                legend_handles.append(
                    plt.Line2D(
                        [0],
                        [0],
                        marker="o",
                        color="w",
                        markerfacecolor=colors[i],
                        markersize=10,
                    )
                )
                legend_labels.append(f"Seq {seq_id}")
            for node in G.nodes:
                seq_id = G.nodes[node]["seq_id"]
                node_color_map[node] = colors[seq_id]
        elif color_by_key == "head":
            unique_head_ids = set(G.nodes[node]["head_id"] for node in G.nodes)
            for i, head_id in enumerate(sorted(list(unique_head_ids))):
                legend_handles.append(
                    plt.Line2D(
                        [0],
                        [0],
                        marker="o",
                        color="w",
                        markerfacecolor=colors[i],
                        markersize=10,
                    )
                )
                legend_labels.append(f"Head {head_id}")
            for node in G.nodes:
                head_id = G.nodes[node]["head_id"]
                node_color_map[node] = colors[head_id]
        elif color_by_key == "node_type":
            for node in G.nodes:
                node_type = G.nodes[node]["node_type"]
                if node_type == "W":
                    node_color_map[node] = colors[0]
                elif node_type == "I":
                    node_color_map[node] = colors[1]
                elif node_type == "O":
                    node_color_map[node] = colors[2]
            legend_handles = [
                plt.Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    markerfacecolor=colors[0],
                    markersize=10,
                ),
                plt.Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    markerfacecolor=colors[1],
                    markersize=10,
                ),
                plt.Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    markerfacecolor=colors[2],
                    markersize=10,
                ),
            ]
            legend_labels = ["Work", "Input", "Output"]
        else:
            raise ValueError(f"Unknown color_by_key {color_by_key}")

        fig, ax = plt.subplots()
        # draw nodes
        for node in G.nodes:
            pos = node_pos[node]
            color = node_color_map[node]
            # ax.text(pos[0], pos[1], label, fontsize=8, ha='center', va='center')
            ax.scatter(pos[0], pos[1], color=color, s=2)
        # draw edges
        for edge in G.edges:
            pos1 = node_pos[edge[0]]
            pos2 = node_pos[edge[1]]
            ax.plot(
                [pos1[0], pos2[0]],
                [pos1[1], pos2[1]],
                color="black",
                linewidth=0.01,
                alpha=0.2,
            )
        # draw legend
        ax.legend(legend_handles, legend_labels, loc="best")
        fig.savefig(
            os.path.join(out_dir, f"workload_spec_{color_by_key}.pdf"),
            bbox_inches="tight",
        )
