from typing import Any, List

import numpy as np
from sklearn.cluster import KMeans


class KMeansModel:

    def __init__(
        self,
        n_clusters: int,
        max_iter: int = 300,
        tol: float = 1e-4,
        random_state: int = 42,
    ) -> None:
        """
        Initialize K-Means model

        :param n_clusters: The number of clusters to form.
        :type n_clusters: int
        :param max_iter: Maximum number of iterations of the k-means algorithm for a single run.
        :type max_iter: int
        :param tol: Relative tolerance with regards to Frobenius norm of the difference in the cluster centers of two consecutive iterations to declare convergence.
        :type tol: float
        :param random_state: Determines random number generation for centroid initialization.
        :type random_state: int
        """
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.model = KMeans(
            n_clusters=self.n_clusters,
            max_iter=self.max_iter,
            tol = self.tol,
            random_state=self.random_state
        )

    def fit(self, X: List) -> None:
        """
        Fit the K-Means model on training data

        :param X: The features to cluster
        :type X: list
        """
        self.model.fit(X)

    def predict(self, X: List) -> np.ndarray:
        """
        Predict the closest cluster each sample in X belongs to.

        :param X: The passed features to perform inference
        :type X: list or np.ndarray
        :return: Cluster labels
        :rtype: np.ndarray
        """
        return self.model.predict(X)

    def transform(self, X: List) -> np.ndarray:
        """
        Transform X to a cluster-distance space.

        :param X: The passed features to transform
        :type X: list or np.ndarray
        :return: Distances to cluster centers
        :rtype: np.ndarray
        """
        return self.model.transform(X)

    @property
    def labels_(self) -> np.ndarray:
        """Get labels of each point from the fitted model."""
        return self.model.labels_

    @property
    def cluster_centers_(self) -> np.ndarray:
        """Get the cluster centers."""
        return self.model.cluster_centers_

    def get_model(self) -> Any:
        """Returns the KMeans model object"""
        return self.model
