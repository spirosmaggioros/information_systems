import os

import networkx as nx
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx


def ds_to_graphs(dataset_folder: str) -> dict:
    """
    Transforms raw .txt graph data to networkx graphs(and more)
    :param dataset_folder: Path from current directory to dataset directory(for example: data/ENZYMES)
    :type dataset_fodler: str

    :returns: dict
              graphs: list[nx.Graph]
              graph_classes: list[int]
              node_labels: list[int]
              node_to_graph: dict
              edges: list[[int, int]]

    Example
    _______

    from dataloader.dataloader import ds_to_graphs
    data = ds_to_graphs("data/ENZYMES")
    graphs = data["graphs"]
    """

    no_graphs: int = 0
    graph_classes: list = []
    node_to_graph: dict = {}
    node_labels: list = []
    edges: list = []

    for file in os.listdir(dataset_folder):
        file_path = os.path.join(dataset_folder, file)

        if "_A.txt" in file:
            with open(file_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        parts = line.split(",")
                        if len(parts) == 2:
                            edges.append((int(parts[0].strip()), int(parts[1].strip())))

        if "_graph_labels.txt" in file:
            with open(file_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        no_graphs += 1
                        graph_classes.append(int(line))

        if "_graph_indicator.txt" in file:
            with open(file_path, "r") as f:
                for i, line in enumerate(f):
                    line = line.strip()
                    if line:
                        graph_id = int(line)
                        node_to_graph[(i + 1)] = graph_id

        if "_node_labels.txt" in file:
            with open(file_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        node_labels.append(int(line))

    graphs = [nx.Graph() for _ in range(no_graphs)]
    for edge in edges:
        u, v = edge
        assert node_to_graph[u] == node_to_graph[v], "Nodes existing in multiple graphs"
        graphs[node_to_graph[u] - 1].add_edge(u, v)

    return {
        "graphs": graphs,
        "graph_classes": graph_classes,
        "node_labels": node_labels,
        "node_to_graph": node_to_graph,
        "edges": edges,
    }


def nx_to_torch_data(data: list[nx.Graph]) -> list[Data]:
    """
    Transforms a list of nx.Graph to a list of torch_geometric.data.Data for GNN training
    :param data: the input list of graphs
    :type data: list[nx.Graph]

    :returns: list[torch_geometric.data.Data]

    Example
    _______
    from dataloader.dataloader import ds_to_graphs
    data = ds_to_graphs("data/ENZYMES")
    graphs = data["graphs"]

    gnn_data = nx_to_torch_data(graphs)
    """
    torch_data = []
    for graph in data:
        torch_data.append(from_networkx(graph))

    return torch_data
