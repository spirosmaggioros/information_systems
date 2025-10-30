from typing import List, Optional

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import networkx as nx


def plot_graph(graph: nx.Graph, node_labels: Optional[List[int]] = None) -> None:
    """
    Simple plot of a nx.Graph with node labels
    - If no node_labels are given: nodes are blue and labeled by ID
    - If node_labels are given: each label has a unique color
    - Edges colored by their 'edge_label' attribute if present, otherwise gray

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
    edges = list(graph.edges())
    pos = nx.spring_layout(graph, seed=42)

    if node_labels:
        curr_node_labels = [node_labels[n - 1] for n in nodes]
        unique_labels = sorted(set(curr_node_labels))
        color_map = cm.get_cmap("tab10", len(unique_labels))
        label_to_color = {lbl: color_map(i) for i, lbl in enumerate(unique_labels)}
        node_colors = [label_to_color[lbl] for lbl in curr_node_labels]
    else:
        node_colors = ["skyblue"] * len(nodes)

    labels = {n: str(n) for n in nodes}

    edge_attrs = nx.get_edge_attributes(graph, "edge_label")
    if edge_attrs:
        unique_edge_labels = sorted(set(edge_attrs.values()))
        edge_cmap = cm.get_cmap("Set2", len(unique_edge_labels))
        edge_label_to_color = {
            lbl: edge_cmap(i) for i, lbl in enumerate(unique_edge_labels)
        }
        edge_colors = [edge_label_to_color[edge_attrs[e]] for e in edges]
    else:
        edge_colors = ["gray"] * len(edges)

    nx.draw(
        graph,
        pos,
        with_labels=True,
        labels=labels,
        node_color=node_colors,
        edge_color=edge_colors,
        node_size=600,
        font_size=9,
    )

    if node_labels:
        for lbl, color in label_to_color.items():
            plt.scatter([], [], color=color, label=f"Node Label {lbl}")
        plt.legend(title="Labels", loc="upper left")

    if edge_attrs:
        for lbl, color in edge_label_to_color.items():
            plt.plot([], [], color=color, label=f"Edge Label {lbl}", linewidth=3)
        plt.legend(title="Labels", loc="upper right")

    plt.show()
