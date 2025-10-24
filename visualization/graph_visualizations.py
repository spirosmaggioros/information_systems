import matplotlib.pyplot as plt
import networkx as nx


def plot_graph(graph: nx.Graph, node_labels: list = []) -> None:
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

    nodes = graph.nodes()

    curr_node_labels = []
    for n in nodes:
        curr_node_labels.append(node_labels[n - 1])
    labels = dict(zip(graph.nodes(), curr_node_labels))

    pos = nx.spring_layout(graph)
    nx.draw(graph, pos, with_labels=True, labels=labels)
    plt.show()
