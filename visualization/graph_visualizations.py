import matplotlib.pyplot as plt
import networkx as nx
import matplotlib.cm as cm


def plot_graph(graph: nx.Graph, node_labels: list = None) -> None:
    """
    Simple plot of a nx.Graph with node labels
    - If no node_labels are given: nodes are blue and labeled by ID
    - If node_labels are given: each label has a unique color

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

    if not node_labels:
        labels = {n: str(n) for n in nodes}
        node_colors = "skyblue"
        nx.draw(
            graph,
            pos,
            with_labels=True,
            labels=labels,
            node_color=node_colors,
            node_size=600,
            font_size=9,
            edge_color="gray",
        )

    else:
        curr_node_labels = [node_labels[n - 1] for n in nodes]

        unique_labels = sorted(set(curr_node_labels))
        color_map = cm.get_cmap("tab10", len(unique_labels))
        label_to_color = {lbl: color_map(i) for i, lbl in enumerate(unique_labels)}
        node_colors = [label_to_color[lbl] for lbl in curr_node_labels]

        labels = {n: str(n) for n in nodes}
        nx.draw(
            graph,
            pos,
            with_labels=True,
            labels=labels,
            node_color=node_colors,
            node_size=600,
            font_size=9,
            edge_color="gray",
        )

        for lbl, color in label_to_color.items():
            plt.scatter([], [], color=color, label=f"Label {lbl}")
        plt.legend(title="Node Labels")

    plt.show()