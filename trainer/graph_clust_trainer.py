import ast
from typing import Any, Dict, List, Tuple, Union

import networkx as nx
import numpy as np
import optuna

from ml_models.graph_models.graph2vec import Graph2Vec
from ml_models.graph_models.netLSD import NetLSD
from trainer.clustering_trainer import train as train_clusterer
from trainer.utils import convert_embeddings_to_real


def train(
    graph_model: str,
    graphs: List[nx.Graph],
    labels: List[int],
    num_classes: int,
    cluster_model: str,
) -> Tuple[Union[Graph2Vec, NetLSD], Dict[str, Any]]:
    """
    Finds the best graph embedding hyperparameters for a downstream clustering task.

    :param graph_model: 'graph2vec' or 'netlsd'
    :type graph_model: str
    :param graphs: List of NetworkX graphs
    :type graphs: List[nx.Graph]
    :param labels: Graph labels (for ARI scoring)
    :type labels: List[int]
    :param num_classes: The number of clusters (fixed 'k')
    :type num_classes: int
    :param cluster_model: 'kmeans' or 'spectral'
    :type cluster_model: str

    :returns: Trained unsupervised model, classifier, and metrics
    :rtype: Tuple[model, Dict[str, float]]

    Example
    _______
    from trainer.graph_clust_trainer import train as train_best_models
    from dataloader.dataloader import ds_to_graphs


    data = ds_to_graphs("data/IMDB-MULTI")

    graphs=data["graphs"]
    labels=data["graph_classes"]

    model, metrics = train_best_models(
        graph_model="netlsd",
        graphs=graphs,
        labels=labels,
        num_classes=6,
        cluster_model="kmeans"
    )
    print(f"Final accuracy: {metrics['ARI']:.4f}")
    """
    assert len(graphs) == len(labels), "Number of graphs must match number of labels"
    assert len(graphs) > 0, "Graph list cannot be empty"

    def objective(trial: Any) -> float:
        if graph_model == "graph2vec":
            wl_iterations = trial.suggest_categorical(
                "wl_iterations", [x for x in range(2, 4)]
            )
            dimensions = trial.suggest_categorical(
                "dimensions", [x for x in range(64, 256, 32)]
            )
            learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-1, log=True)

            model_g2v = Graph2Vec(
                wl_iterations=wl_iterations,
                dimensions=dimensions,
                learning_rate=learning_rate,
            )
            model_g2v.fit(graphs)
            embeddings = model_g2v.get_embeddings()
        else:
            kernel = trial.suggest_categorical("kernel", ["heat", "wave"])
            num_timescales = trial.suggest_categorical(
                "num_timescales", [64, 128, 250, 400]
            )

            if kernel == "heat":
                t_min = trial.suggest_float("t_min", 1e-3, 1e-1, log=True)
                t_max = trial.suggest_float("t_max", 10, 100, log=True)
                timescales = np.logspace(
                    np.log10(t_min), np.log10(t_max), num_timescales
                )
            else:
                max_t = trial.suggest_float("max_t", np.pi, 2 * np.pi)
                timescales = np.linspace(0, max_t, num_timescales)

            eigenvalues_choice = trial.suggest_categorical(
                "eigenvalues",
                [
                    "auto",
                    "full",
                    2,
                    3,
                    4,
                    5,
                    6,
                    7,
                    8,
                    9,
                    10,
                    12,
                    16,
                    18,
                    20,
                    28,
                    32,
                    40,
                    50,
                    64,
                    80,
                    100,
                    "(2, 4)",
                    "(4, 2)",
                    "(6, 2)",
                    "(2, 6)",
                    "(8, 4)",
                    "(4, 8)",
                    "(10, 20)",
                    "(20, 10)",
                    "(32, 4)",
                    "(4, 32)",
                    "(10, 50)",
                    "(50, 10)",
                ],
            )

            eigenvalues_param: Union[str, int, Tuple[int, int]]
            if isinstance(eigenvalues_choice, str) and eigenvalues_choice.startswith(
                "("
            ):
                eigenvalues_param = ast.literal_eval(eigenvalues_choice)
            else:
                eigenvalues_param = eigenvalues_choice

            normalization = trial.suggest_categorical(
                "normalization", ["empty", "complete", None]
            )
            normalized_laplacian = trial.suggest_categorical(
                "normalized_laplacian", [True, False]
            )

            model_netlsd = NetLSD(
                timescales=timescales,
                kernel=kernel,
                eigenvalues=eigenvalues_param,
                normalization=normalization,
                normalized_laplacian=normalized_laplacian,
            )

            try:
                embeddings_raw = [model_netlsd.fit_transform(g) for g in graphs]
                embeddings = convert_embeddings_to_real(embeddings_raw)
            except Exception as e:
                trial.set_user_attr("Failed due to wrong hyperparameters", str(e))
                return float("-inf")

        _, stats = train_clusterer(
            graph_embeddings=embeddings,
            labels=labels,
            num_classes=num_classes,
            model_type=cluster_model,
        )
        checkpoint = {
            "trial_params": trial.params,
            "ARI": stats["ARI"],
        }
        trial.set_user_attr("checkpoint", checkpoint)

        return float(stats["ARI"])

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=100)

    best_trial = study.best_trial
    best_checkpoint = best_trial.user_attrs["checkpoint"]
    best_hyperparams = best_checkpoint["trial_params"]
    best_model: Union[Graph2Vec, NetLSD]

    if graph_model == "graph2vec":
        best_model_g2v = Graph2Vec(
            wl_iterations=best_hyperparams["wl_iterations"],
            dimensions=best_hyperparams["dimensions"],
            learning_rate=best_hyperparams["learning_rate"],
        )
        best_model_g2v.fit(graphs)
        embeddings = best_model_g2v.get_embeddings()
        best_model = best_model_g2v
    else:
        eigenvalues_choice = best_hyperparams["eigenvalues"]
        eigenvalues_param: Union[str, int, Tuple[int, int]]
        if isinstance(eigenvalues_choice, str) and eigenvalues_choice.startswith("("):
            eigenvalues_param = ast.literal_eval(eigenvalues_choice)
        else:
            eigenvalues_param = eigenvalues_choice

        best_model_netlsd = NetLSD(
            timescales=best_hyperparams["timescales"],
            kernel=best_hyperparams["kernel"],
            eigenvalues=eigenvalues_param,
            normalization=best_hyperparams["normalization"],
            normalized_laplacian=best_hyperparams["normalized_laplacian"],
        )
        embeddings_raw = [best_model_netlsd.fit_transform(g) for g in graphs]
        embeddings = convert_embeddings_to_real(embeddings_raw)
        best_model = best_model_netlsd

    _, final_stats = train_clusterer(
        graph_embeddings=embeddings,
        labels=labels,
        num_classes=num_classes,
        model_type=cluster_model,
    )

    return best_model, final_stats
