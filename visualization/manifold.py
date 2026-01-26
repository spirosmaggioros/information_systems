import matplotlib.pyplot as plt
import numpy as np
import umap
import umap.plot
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, Isomap


def visualize_embeddings_manifold(
    features: list,
    labels: list,
    method: str = "TSNE",
    n_components: int = 2,
    save_to: str = "manifold_visualization.png",
) -> None:
    """
    Fit a manifold learning model on passed features. Labels are given
    only for visualization purposes.

    :param features: Passed features manifold learning methods will be fitted on
    :type features: list
    :param labels: Ground truth labels for each feature
    :type labels: list
    :param method: Either TSNE, Isomap or UMAP
    :type method: str
    :param n_components: Latent dimensions of manifold methods
    :type n_components: int(use 2 or 3)
    :param save_to: Complete path of the output png plot
    :type save_to: str
    """
    assert method in ["TSNE", "Isomap", "UMAP"]
    assert len(features) == len(labels)
    latent_features: list = []
    if method == "TSNE":
        model = TSNE(n_components=n_components)
    elif method == "Isomap":
        model = Isomap(n_components=n_components)
    else:
        model = umap.UMAP(n_neighbors=150, min_dist=0.2)

    if not isinstance(model, umap.UMAP) and len(features[0]) > 50:
        pca_features = PCA(n_components=50).fit_transform(features)
        latent_features = model.fit_transform(pca_features)
    else:
        if isinstance(model, umap.UMAP):
            model.fit(features)
        else:
            latent_features = model.fit_transform(features)

    if n_components == 2:
        if isinstance(model, umap.UMAP):
            umap.plot.points(model, labels=np.array(labels))
        else:
            scatter = plt.scatter(
                [x[0] for x in latent_features],
                [x[1] for x in latent_features],
                c=labels,
                cmap="tab10",
            )
            plt.xlabel("X")
            plt.ylabel("Y")
            plt.grid(True)
            plt.title(f"Latent embeddings with {method}")

            cbar = plt.colorbar(scatter)
            cbar.set_label("Labels")
    else:
        if isinstance(model, umap.UMAP):
            print("3D manifold for UMAP is not available!")
            exit(0)

        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")
        scatter = ax.scatter(
            [x[0] for x in latent_features],
            [x[1] for x in latent_features],
            [x[2] for x in latent_features],
            c=labels,
            cmap="tab10",
        )

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.grid(True)
        ax.set_title(f"Latent embeddings with {method}")

        fig.colorbar(scatter, ax=ax, label="Labels")

    plt.savefig(save_to)
    plt.show()
