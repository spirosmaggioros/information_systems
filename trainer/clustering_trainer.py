from typing import Any, List, Type, Union

import numpy as np
import optuna
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import StandardScaler

from ml_models.clustering.kmeans import KMeansModel
from ml_models.clustering.spectral_clustering import SpectralClusteringModel


def train(
    graph_embeddings: List,
    labels: List,
    num_classes: int,
    model_type: str = "kmeans",
) -> Any:
    """
    Trains a clustering model using Optuna for perparameter optimization.

    The objective is to maximize the Adjusted Rand Index (ARI) using 5-fold cross-validation.

    :param graph_embeddings: Graph embeddings.
    :type graph_embeddings: List
    :param labels: True labels for scoring (ARI).
    :type labels: List
    :param num_classes: The number of clusters (fixed 'k').
    :type num_classes: int
    :param model_type: 'kmeans' or 'spectral'.
    :type model_type: str
    :return: A tuple of (best_model, stats_dictionary)
    :rtype: Any
    """

    X = np.array(graph_embeddings)
    y = np.array(labels)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    def objective(trial: Any) -> float:
        ModelClass: Union[Type[KMeansModel], Type[SpectralClusteringModel]]
        model_params: dict[str, Any] = {"n_clusters": num_classes, "random_state": 42}

        if model_type == "kmeans":
            model_params["max_iter"] = trial.suggest_categorical(
                "max_iter", [100, 300, 500]
            )
            model_params["tol"] = trial.suggest_float("tol", 1e-5, 1e-3, log=True)
            ModelClass = KMeansModel

        elif model_type == "spectral":
            model_params["affinity"] = trial.suggest_categorical(
                "affinity", ["rbf", "nearest_neighbors"]
            )
            if model_params["affinity"] == "rbf":
                model_params["gamma"] = trial.suggest_float("gamma", 1.0, 1e2, log=True)
            else:
                model_params["n_neighbors"] = trial.suggest_int(
                    "n_neighbors",
                    3,
                    10,
                )
            ModelClass = SpectralClusteringModel
        else:
            raise ValueError("Unsupported model_type")

        model = ModelClass(**model_params)
        model.fit(X_scaled)
        y_pred = model.predict(X_scaled)
        ari = adjusted_rand_score(y_pred, y)

        trial.set_user_attr("checkpoint", {"mean_ARI": ari})
        return float(ari)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=50)

    best_trial = study.best_trial
    best_hyperparams = best_trial.params

    final_params: dict[str, Any] = {
        "n_clusters": num_classes,
        "random_state": 42,
        **best_hyperparams,
    }

    best_model: Union[KMeansModel, SpectralClusteringModel]
    if model_type == "kmeans":
        best_model = KMeansModel(**final_params)
    else:
        best_model = SpectralClusteringModel(**final_params)

    best_model.fit(X_scaled)

    if model_type == "kmeans":
        final_labels = best_model.labels_
    else:
        final_labels = best_model.predict(X_scaled)

    final_ari = adjusted_rand_score(y, final_labels)

    stats = {
        "ARI": final_ari,
    }

    return best_model, stats
