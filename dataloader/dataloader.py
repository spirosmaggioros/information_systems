import os
from collections import defaultdict

import networkx as nx


def ds_to_graphs(dataset_folder: str) -> dict:
    no_graphs: int = 0
    classes: list = []
    graph_to_nodes: dict = defaultdict(list)
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
                        classes.append(int(line))

        if "_graph_indicator.txt" in file:
            with open(file_path, "r") as f:
                for i, line in enumerate(f):
                    line = line.strip()
                    if line:
                        graph_id = int(line)
                        graph_to_nodes[graph_id].append(i + 1)

        if "_node_labels.txt" in file:
            with open(file_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        node_labels.append(int(line))

    graphs = [nx.Graph() for _ in range(no_graphs)]

    return {
        "graphs": graphs,
        "classes": classes,
        "node_labels": node_labels,
        "graph_to_nodes": dict(graph_to_nodes),
        "edges": edges,
    }
