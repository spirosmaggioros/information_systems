from typing import Any, Dict

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.patches import FancyArrowPatch


def _draw_curved_edge(
    ax: plt.Axes,
    pos: Dict[Any, np.ndarray],
    u: Any,
    v: Any,
    rad: float,
    color: str = "gray",
    linewidth: float = 1.5,
    alpha: float = 0.9,
) -> None:
    """Draw a single curved edge between u and v using FancyArrowPatch."""
    patch = FancyArrowPatch(
        posA=pos[u],
        posB=pos[v],
        connectionstyle=f"arc3,rad={rad}",
        arrowstyle="-",
        linewidth=linewidth,
        alpha=alpha,
        color=color,
        shrinkA=0,
        shrinkB=0,
        mutation_scale=10,
    )
    ax.add_patch(patch)


def plot_graph(
    graph: nx.Graph | nx.MultiGraph, node_labels: list = [], max_rad: float = 0.2
) -> None:
    """
    Simple plot of a nx.Graph with node labels

    :param graph: The input graph
    :type graph: nx.Graph
    :param node_labels: Optionally pass the node labels
    :type node_labels: list[int]

    Example
    _______

    from dataloader.dataloader import ds_to_graphs
    from visualization.graph_visualizations import plot_graph

    data = ds_to_graphs("data/ENZYMES")
    graphs = data["graphs"]

    plot_graph(graphs[0], data["node_labels"])
    """

    nodes = list(graph.nodes())
    pos = nx.spring_layout(graph, seed=42)

    if node_labels:
        curr_node_labels = [node_labels[n] for n in nodes]
        unique_labels = sorted(set(curr_node_labels))
        color_map = cm.get_cmap("tab10", len(unique_labels))
        label_to_color = {lbl: color_map(i) for i, lbl in enumerate(unique_labels)}
        node_colors = [label_to_color[lbl] for lbl in curr_node_labels]
    else:
        node_colors = ["skyblue"] * len(nodes)

    labels = {n: str(n) for n in nodes}

    edge_attrs = nx.get_edge_attributes(graph, "edge_label")
    edge_label_to_color = {}
    if edge_attrs:
        unique_edge_labels = sorted(set(edge_attrs.values()))
        edge_cmap = cm.get_cmap("Set2", len(unique_edge_labels))
        edge_label_to_color = {
            lbl: edge_cmap(i) for i, lbl in enumerate(unique_edge_labels)
        }

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_axis_off()

    if isinstance(graph, nx.MultiGraph):
        edge_dict: Dict[tuple, list] = {}
        for u, v, key in graph.edges(keys=True):
            pair = tuple(sorted((u, v)))
            edge_dict.setdefault(pair, []).append(key)

        for (u, v), keys in edge_dict.items():
            num_to_draw = max(1, len(keys) // 2)

            rad_list = (
                [0.0]
                if num_to_draw == 1
                else np.linspace(-max_rad, max_rad, num_to_draw)
            )

            for rad in rad_list:
                if edge_attrs and (u, v) in edge_attrs:
                    color = edge_label_to_color[edge_attrs[(u, v)]]
                else:
                    color = "gray"
                _draw_curved_edge(
                    ax, pos, u, v, rad=rad, color=color, linewidth=1.5, alpha=0.9
                )
    else:
        for u, v in graph.edges():
            color = (
                edge_label_to_color[edge_attrs[(u, v)]]
                if edge_attrs and (u, v) in edge_attrs
                else "gray"
            )
            _draw_curved_edge(
                ax, pos, u, v, rad=0.0, color=color, linewidth=1.5, alpha=0.9
            )

    nx.draw_networkx_nodes(graph, pos, node_color=node_colors, node_size=600, ax=ax)
    nx.draw_networkx_labels(graph, pos, labels=labels, font_size=9, ax=ax)

    if node_labels:
        for lbl, color in label_to_color.items():
            plt.scatter([], [], color=color, label=f"Node Label {lbl}")
        plt.legend(title="Node Labels", loc="upper left")

    if edge_attrs:
        for lbl, color in edge_label_to_color.items():
            plt.plot([], [], color=color, label=f"Edge Label {lbl}", linewidth=3)
        plt.legend(title="Edge Labels", loc="upper right")

    plt.tight_layout()
    plt.show()
