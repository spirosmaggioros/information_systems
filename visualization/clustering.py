from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA


def scatter_clusters(
    model: Any, data: list, title: str = "Clustering Results", figsize: tuple = (10, 8)
) -> None:
    """
    Create a scatter plot of clustering results.

    :param model: Trained clustering model (KMeans, SpectralClustering, etc.)
    :type model: Any
    :param data: Input data points
    :type data: list
    """

    cluster_labels = model.predict(data) if hasattr(model, "predict") else model.labels_

    if len(data[0]) > 2:
        pca = PCA(n_components=2)
        data_2d = pca.fit_transform(data)
    else:
        data_2d = np.array(data)

    plt.figure(figsize=figsize)

    unique_clusters = np.unique(cluster_labels)
    colors = plt.cm.Set1(np.linspace(0, 1, len(unique_clusters)))

    for i, cluster in enumerate(unique_clusters):
        cluster_mask = cluster_labels == cluster
        plt.scatter(
            data_2d[cluster_mask, 0],
            data_2d[cluster_mask, 1],
            c=[colors[i]],
            label=f"Cluster {cluster}",
            alpha=0.7,
            s=50,
        )

    plt.xlabel("Feature 1" if len(data[0]) >= 2 else "PC1")
    plt.ylabel("Feature 2" if len(data[0]) >= 2 else "PC2")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
