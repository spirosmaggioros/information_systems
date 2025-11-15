import os
import random
import re
from itertools import combinations
from typing import Any, List, Tuple

import networkx as nx
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
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
              node_attributes: dict
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
    node_attributes: dict = {}
    edges: list = []
    edge_labels: list = []
    is_multigraph = False
    seen_edges = set()

    for file in os.listdir(dataset_folder):
        file_path = os.path.join(dataset_folder, file)

        if "_A.txt" in file:
            with open(file_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        parts = line.split(",")
                        edge = (int(parts[0].strip()), int(parts[1].strip()))
                        edges.append(edge)
                        if edge in seen_edges:
                            is_multigraph = True
                        seen_edges.add(edge)

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

        if "_node_attributes.txt" in file:
            with open(file_path, "r") as f:
                for i, line in enumerate(f):
                    line = line.strip()
                    attr = [float(x) for x in re.split("[,]", line)]
                    node_attributes[(i + 1)] = attr

        if "_node_labels.txt" in file:
            with open(file_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        node_labels.append(int(line))

        if "_edge_labels.txt" in file:
            with open(file_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        edge_labels.append(int(line))

    if is_multigraph:
        graphs = [nx.MultiGraph() for _ in range(no_graphs)]
    else:
        graphs = [nx.Graph() for _ in range(no_graphs)]

    has_edge_labels = len(edge_labels) > 0

    for edge in edges:
        u, v = edge
        assert node_to_graph[u] == node_to_graph[v], "Nodes existing in multiple graphs"
        graphs[node_to_graph[u] - 1].add_edge(u, v)

    for g in graphs:
        nx.set_node_attributes(g, node_attributes, name="node_attributes")

    if has_edge_labels:
        edge_attr_dict = {edges[i]: edge_labels[i] for i in range(len(edge_labels))}
        for g in graphs:
            nx.set_edge_attributes(g, edge_attr_dict, name="edge_label")

    for i, graph in enumerate(graphs):
        graphs[i] = nx.convert_node_labels_to_integers(graph, first_label=0)

    return {
        "graphs": graphs,
        "graph_classes": graph_classes,
        "node_labels": node_labels,
        "node_attributes": node_attributes,
        "node_to_graph": node_to_graph,
        "edges": edges,
        "edge_labels": edge_labels,
    }


def nx_to_torch_data(data: List[nx.Graph]) -> List[Data]:
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


class GraphDataloader(Dataset):
    """
    Class to transform a list of graphs to batches for torch training

    :param mode: either train, evaluate or inference
    :param data: input graphs
    :type data: List[nx.Graph]
    :param labels: a list of labels for each graph
    :type labels: list
    """

    def __init__(
        self,
        mode: str,
        data: List[nx.Graph],
        labels: List,
        device: str = "mps",
    ) -> None:
        super().__init__()

        assert mode in ["train", "evaluate", "inference"]

        self.mode = mode
        self.data = nx_to_torch_data(data)
        self.labels = labels
        self.device = device

        if labels is not None:
            self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> Any:
        graph = self.data[index]
        if self.mode != "inference" and self.labels is not None:
            graph.y = self.labels[index]

        # graph.node_attributes = torch.tensor(graph.node_attributes, device=self.device)
        return graph


def create_graph_dataloaders(
    data: List[nx.Graph],
    labels: List,
    batch_size: int,
    test_size: float = 0.2,
    device: str = "mps",
) -> Tuple[DataLoader, DataLoader]:
    """
    Generates train and test dataloaders for graph classification tasks

    :param data: Passed list of graphs
    :type data: List[nx.Graph]
    :param labels: Graph classes
    :type labels: List
    :param batch_size: The input batch size
    :type batch_size: int
    :param test_size: The size of test dataloader
    :type test_size: float(default = 0.2)

    Example
    _______
    data = ds_to_graphs("data/ENZYMES")
    train_dataloader, test_dataloader = create_graph_dataloaders(data["graphs"], data["graph_classes"], 2)
    """
    X_train, X_test, y_train, y_test = train_test_split(
        data,
        labels,
        random_state=42,
        test_size=test_size,
    )
    train_loader = GraphDataloader(
        mode="train", data=X_train, labels=y_train, device=device
    )
    test_loader = GraphDataloader(
        mode="evaluate", data=X_test, labels=y_test, device=device
    )

    train_dataloader = DataLoader(
        train_loader,
        batch_size=batch_size,
        shuffle=True,
        num_workers=os.cpu_count() // 4,  # type: ignore
    )

    test_dataloader = DataLoader(
        test_loader,
        batch_size=batch_size,
        shuffle=False,
        num_workers=os.cpu_count() // 4,  # type: ignore
    )

    return train_dataloader, test_dataloader


def add_random_edges(G: nx.Graph, p: float = 0.2) -> None:
    """
    Adds random edges to G

    :param G: The input graph
    :type G: nx.Graph
    :param p: The % of the initial edges that will be added to G
    :type p: float(default = 0.2)
    """
    init_edges = len(G.edges())
    possible_edges = list(combinations(list(G.nodes()), 2))
    random.shuffle(possible_edges)

    added_edges = 0
    for edge in possible_edges:
        if not G.has_edge(edge[0], edge[1]):
            G.add_edge(edge[0], edge[1])
            added_edges += 1

        if added_edges >= p * init_edges:
            break


def remove_random_edges(G: nx.Graph, p: float = 0.2) -> None:
    """
    Removes random edges from G

    :param G: The input graph
    :type G: nx.Graph
    :param p: The % of the initial edges that will be removed from G
    :type p: float(default = 0.2)
    """
    edges = list(G.edges())
    init_edges = len(edges)

    removed_edges = 0
    for edge in edges:
        if G.has_edge(edge[0], edge[1]):
            G.remove_edge(edge[0], edge[1])
            removed_edges += 1

        if removed_edges >= p * init_edges:
            break


def shuffle_node_attributes(G: nx.Graph) -> None:
    """
    Shuffles node attributes

    :param G: The input graph
    :type G: nx.Graph
    """
    init_node_attributes = nx.get_node_attributes(G, "nøde_attributes")
    nodes = list(init_node_attributes.keys())
    shuffled_nodes = random.sample(list(init_node_attributes.keys()), len(nodes))
    shuffled_node_attributes = {}

    for i, node in enumerate(nodes):
        shuffled_node_attributes[shuffled_nodes[i]] = init_node_attributes[node]

    nx.set_node_attributes(G, shuffled_node_attributes)
