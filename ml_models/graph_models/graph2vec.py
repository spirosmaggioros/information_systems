from typing import List

import networkx as nx
import numpy as np
from karateclub import Graph2Vec as KCGraph2Vec


class Graph2Vec:
    """
    Graph2Vec embedding model for whole-graph representation learning.

    :param wl_iterations: Number of Weisfeiler-Lehman iterations
    :type wl_iterations: int
    :param attributed: Presence of graph attributes
    :type attributed: bool
    :param dimensions: Dimensionality of embedding
    :type dimensions: int
    :param workers: Number of cores
    :type workers: int
    :param down_sampling: Down sampling frequency
    :type down_sampling: float
    :param epochs: Number of epochs
    :type epochs: int
    :param learning_rate: HogWild! learning rate
    :type learning_rate: float
    :param min_count: Minimal count of graph feature occurrences
    :type min_count: int
    :param seed: Random seed for the model
    :type seed: int
    :param erase_base_features: Erasing the base features
    :type erase_base_features: bool

    Example
    _______
    from ml_models.graph_models.graph2vec import Graph2Vec
    from dataloader.dataloader import ds_to_graphs

    data = ds_to_graphs("data/MUTAG")
    model = Graph2Vec(dimensions=128, wl_iterations=2, epochs=100)
    model.fit(data["graphs"])
    embeddings = model.get_embedding()
    """

    def __init__(
        self,
        wl_iterations: int = 2,
        attributed: bool = False,
        dimensions: int = 128,
        workers: int = 4,
        down_sampling: float = 0.0001,
        epochs: int = 10,
        learning_rate: float = 0.025,
        min_count: int = 5,
        seed: int = 42,
        erase_base_features: bool = False,
    ) -> None:
        self.model = KCGraph2Vec(
            wl_iterations=wl_iterations,
            attributed=attributed,
            dimensions=dimensions,
            workers=workers,
            down_sampling=down_sampling,
            epochs=epochs,
            learning_rate=learning_rate,
            min_count=min_count,
            seed=seed,
            erase_base_features=erase_base_features,
        )

    def fit(self, graphs: List[nx.Graph]) -> None:
        """
        Fit the Graph2Vec model.

        :param graphs: List of NetworkX graphs
        :type graphs: List[nx.Graph]
        """
        self.model.fit(graphs)

    def get_embedding(self) -> np.ndarray:
        """
        Get the learned embeddings.

        :returns: Graph embeddings
        :rtype: np.ndarray
        """
        return self.model.get_embedding()

    def infer(self, graphs: List[nx.Graph]) -> np.ndarray:
        """
        Infer embeddings for new graphs.

        :param graphs: List of NetworkX graphs
        :type graphs: List[nx.Graph]
        :returns: Graph embeddings
        :rtype: np.ndarray
        """
        return self.model.infer(graphs)
