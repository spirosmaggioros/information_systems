from typing import List

import networkx as nx
import numpy as np
from karateclub import DeepWalk as KCDeepWalk


class DeepWalk:
    """
    DeepWalk embedding model for graph representation learning.

    :param walk_number: Number of random walks per node
    :type walk_number: int
    :param walk_length: Length of each random walk
    :type walk_length: int
    :param dimensions: Dimensionality of embedding
    :type dimensions: int
    :param workers: Number of cores
    :type workers: int
    :param window_size: Window size for Skip-gram
    :type window_size: int
    :param epochs: Number of epochs
    :type epochs: int
    :param learning_rate: Learning rate
    :type learning_rate: float
    :param min_count: Minimal count of node occurrences
    :type min_count: int
    :param seed: Random seed for the model
    :type seed: int

    Example
    _______
    from ml_models.graph_models.deepwalk import DeepWalk
    from dataloader.dataloader import ds_to_graphs

    data = ds_to_graphs("data/MUTAG")
    model = DeepWalk(dimensions=128, walk_number=10, walk_length=80)
    model.fit(data["graphs"])
    embeddings = model.get_embeddings()
    """

    def __init__(
        self,
        walk_number: int = 10,
        walk_length: int = 80,
        dimensions: int = 128,
        workers: int = 4,
        window_size: int = 5,
        epochs: int = 10,
        learning_rate: float = 0.05,
        min_count: int = 1,
        seed: int = 42,
    ) -> None:
        self.walk_number = walk_number
        self.walk_length = walk_length
        self.dimensions = dimensions
        self.workers = workers
        self.window_size = window_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.min_count = min_count
        self.seed = seed
        self.graph_embeddings = None

    def fit(self, graphs: List[nx.Graph]) -> None:
        """
        Fit the DeepWalk model.

        :param graphs: List of NetworkX graphs
        :type graphs: List[nx.Graph]
        """
        embeddings = []
        for graph in graphs:
            model = KCDeepWalk(
                walk_number=self.walk_number,
                walk_length=self.walk_length,
                dimensions=self.dimensions,
                workers=self.workers,
                window_size=self.window_size,
                epochs=self.epochs,
                learning_rate=self.learning_rate,
                min_count=self.min_count,
                seed=self.seed,
            )
            model.fit(graph)
            node_embeddings = model.get_embedding()
            graph_embedding = np.mean(node_embeddings, axis=0)
            embeddings.append(graph_embedding)

        self.graph_embeddings = np.array(embeddings)

    def get_embeddings(self) -> np.ndarray:
        """
        Get the learned embeddings.

        :returns: Graph embeddings
        :rtype: np.ndarray
        """
        return self.graph_embeddings

    def infer(self, graphs: List[nx.Graph]) -> np.ndarray:
        """
        Infer embeddings for new graphs.

        :param graphs: List of NetworkX graphs
        :type graphs: List[nx.Graph]
        :returns: Graph embeddings
        :rtype: np.ndarray
        """
        embeddings = []
        for graph in graphs:
            model = KCDeepWalk(
                walk_number=self.walk_number,
                walk_length=self.walk_length,
                dimensions=self.dimensions,
                workers=self.workers,
                window_size=self.window_size,
                epochs=self.epochs,
                learning_rate=self.learning_rate,
                min_count=self.min_count,
                seed=self.seed,
            )
            model.fit(graph)
            node_embeddings = model.get_embedding()
            graph_embedding = np.mean(node_embeddings, axis=0)
            embeddings.append(graph_embedding)

        return np.array(embeddings)
