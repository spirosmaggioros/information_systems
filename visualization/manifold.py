import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, Isomap


def visualize_embeddings_manifold(
    features: list,
    labels: list,
    method: str = "TSNE",
    n_components: int = 2,
    save_to: str = "tsne_visualization.png",
) -> None:
    assert method in ["TSNE", "Isomap", "UMAP"]
    assert len(features) == len(labels)
    latent_features: list = []
    if method == "TSNE":
        model = TSNE(n_components=n_components)
    elif method == "Isomap":
        model = Isomap(n_components=n_components)
    else:
        pass

    if len(features[0]) > 30:
        pca_features = PCA(n_components=30).fit_transform(features)
        latent_features = model.fit_transform(pca_features)
    else:
        latent_features = model.fit_transform(features)

    if n_components == 2:
        plt.scatter(
            [x[0] for x in latent_features],
            [x[1] for x in latent_features],
            c=labels,
        )
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.title(f"Latent embeddings with {method}")
    else:
        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")
        ax.scatter(
            [x[0] for x in latent_features],
            [x[1] for x in latent_features],
            [x[2] for x in latent_features],
            c=labels,
        )
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title(f"Latent embeddings with {method}")

    plt.savefig(save_to)
    plt.show()
