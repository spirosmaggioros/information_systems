from typing import List

import networkx as nx
import numpy as np
from karateclub import NetLSD as KCNetLSD


class NetLSD:
    """
    NetLSD embedding model for whole-graph representation learning.

    :param scale_min: Time scale interval minimum
    :type scale_min: float
    :param scale_max: Time scale interval maximum
    :type scale_max: float
    :param scale_steps: Number of steps in time scale
    :type scale_steps: int
    :param approximations: Number of eigenvalue approximations
    :type approximations: int
    :param seed: Random seed value
    :type seed: int

    Example
    _______
    from ml_models.graph_models.netLSD import NetLSD
    from dataloader.dataloader import ds_to_graphs

    data = ds_to_graphs("data/MUTAG")
    model = NetLSD(scale_min=-2.0, scale_max=2.0, scale_steps=150)
    model.fit(data["graphs"])
    embeddings = model.get_embeddings()
    """

    def __init__(
        self,
        scale_min: float = -2.0,
        scale_max: float = 2.0,
        scale_steps: int = 250,
        approximations: int = 200,
        seed: int = 42,
    ) -> None:
        self.model = KCNetLSD(
            scale_min=scale_min,
            scale_max=scale_max,
            scale_steps=scale_steps,
            approximations=approximations,
            seed=seed,
        )

    def fit(self, graphs: List[nx.Graph]) -> None:
        """
        Fitting a NetLSD model.

        :param graphs: List of NetworkX graphs
        :type graphs: List[nx.Graph]
        """
        self.model.fit(graphs)

    def get_embeddings(self) -> np.ndarray:
        """
        Get the learned embeddings.

        :returns: Graph embeddings
        :rtype: np.ndarray
        """
        return self.model.get_embedding()

    def infer(self, graphs: List[nx.Graph]) -> np.ndarray:
        """
        Inferring the NetLSD embeddings.

        :param graphs: List of NetworkX graphs
        :type graphs: List[nx.Graph]
        :returns: Graph embeddings
        :rtype: np.ndarray
        """
        return self.model.infer(graphs)
