from typing import Any, List

import numpy as np
from sklearn.cluster import SpectralClustering


class SpectralClusteringModel:

    def __init__(
        self,
        n_clusters: int,
        affinity: str = "rbf",
        gamma: float = 1.0,
        n_neighbors: int = 10,
        random_state: int = 42,
    ) -> None:
        """
        Initialize Spectral Clustering model

        :param n_clusters: The number of clusters to form.
        :type n_clusters: int
        :param affinity: How to construct the affinity matrix.
        :type affinity: str (default='rbf')
        :param gamma: Kernel coefficient for rbf. Ignored for affinity='nearest_neighbors'
        :type gamma: float (default='1.0')
        :param n_neighbors: Number of neighbors to use when constructing the affinity matrix using the nearest neighbors method. Ignored for affinity='rbf'.
        type n_neighbors: int
        :param random_state: Determines random number generation.
        :type random_state: int (default=42)
        """
        self.n_clusters = n_clusters
        self.affinity = affinity
        self.gamma = gamma
        self.n_neighbors = n_neighbors
        self.random_state = random_state
        self.model = SpectralClustering(
            n_clusters=self.n_clusters,
            affinity=self.affinity,
            gamma=self.gamma,
            n_neighbors=self.n_neighbors,
            random_state=self.random_state,
            assign_labels="cluster_qr",
        )

    def fit(self, X: List) -> None:
        """
        Fit the Spectral Clustering model on training data

        :param X: The features to cluster
        :type X: list or np.ndarray
        """
        self.model.fit(X)

    def predict(self, X: List) -> np.ndarray:
        """
        Predict the closest cluster each sample in X belongs to.

        :param X: The passed features to perform inference
        :type X: list
        :return: Cluster labels
        :rtype: np.ndarray
        """
        return self.model.fit_predict(X)

    @property
    def labels_(self) -> np.ndarray:
        """Get labels of each point from the fitted model."""
        return self.model.labels_

    def get_model(self) -> Any:
        """Returns the SpectralClustering model object"""
        return self.model
