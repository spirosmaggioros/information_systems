import ast
from typing import Any, Dict, List, Tuple, Union

import networkx as nx
import numpy as np
import optuna
import torch
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader
import tracemalloc

from ml_models.classification.mlp import MLP, MLPDataset
from ml_models.graph_models.deepwalk import DeepWalk
from ml_models.graph_models.graph2vec import Graph2Vec
from ml_models.graph_models.netLSD import NetLSD
from trainer.dl_trainer import train as train_dl
from trainer.svm_trainer import train as train_svm
from trainer.utils import convert_embeddings_to_real


def train_complete_classifier(
    X_train: list,
    X_test: list,
    y_train: list,
    y_test: list,
    num_classes: int,
    mode: str,
    classifier: str = "SVC",
    device: str = "mps",
    save_model: bool = False,
) -> dict:
    """
    Trains the passed classifier on graph embeddings computed by a graph model
    """
    clf = None
    metrics = {"AUROC": 0.0, "F1": 0.0, "Accuracy": 0.0}
    if classifier == "SVC":
        clf, stats = train_svm(mode=mode, graph_embeddings=X_train, labels=y_train)
        metrics["AUROC"] = stats["AUC"]
        metrics["F1"] = stats["F1"]
        metrics["Accuracy"] = stats["Accuracy"]
    else:
        train_ds = MLPDataset(X_train, y_train)
        val_ds = MLPDataset(X_test, y_test)

        train_dataloader = DataLoader(train_ds, batch_size=32, shuffle=True)
        test_dataloader = DataLoader(val_ds, batch_size=32, shuffle=False)

        clf = MLP(
            num_classes=num_classes,
            device=device,
            init_input=len(X_train[0]),
        )
        if mode == "binary":
            loss = nn.BCEWithLogitsLoss()
        else:
            loss = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(
            clf.parameters(),
            lr=0.001,
            weight_decay=0.1,
            amsgrad=True,
        )

        _, best_stats = train_dl(
            model=clf,
            model_type="torch",
            train_dataloader=train_dataloader,
            test_dataloader=test_dataloader,
            num_classes=num_classes,
            loss_fn=loss,
            optimizer=optimizer,
            mode=mode,
            device=device,
            save_model=save_model,
        )

        metrics["AUROC"] = best_stats["test_auc"]
        metrics["F1"] = best_stats["test_f1"]
        metrics["Accuracy"] = best_stats["test_acc"]

    return metrics


def train(
    graph_model: str,
    graphs: List[nx.Graph],
    labels: List[int],
    num_classes: int,
    mode: str,
    out_channels: int = 256,
    epochs: int = 100,
    test_size: float = 0.25,
    classifier: str = "SVC",
    model_name: str = "best_graph_model.pkl",
    device: str = "mps",
) -> Tuple[Union[Graph2Vec, NetLSD, DeepWalk], Dict[str, float]]:
    """
    Train a unsupervised whole-graph model

    :param graph_model: Name of the unsupervised model (graph2vec, netlsd or deepwalk)
    :type graph_model: str
    :param graphs: List of NetworkX graphs
    :type graphs: List[nx.Graph]
    :param labels: Graph labels
    :type labels: List[int]
    :param test_size: Test set size
    :type test_size: float

    :returns: Trained unsupervised model, classifier, and metrics
    :rtype: Tuple[model, Dict[str, float]]

    Example
    _______

    model, metrics = train_best_model(
        graph_model="graph2vec"
        graphs=graphs,
        labels=labels,
        num_classes=3,
    )
    print(f"Final accuracy: {metrics['Accuracy']:.4f}")
    print(f"Final F1: {metrics['F1']:.4f}")
    """
    assert len(graphs) == len(labels), "Number of graphs must match number of labels"
    assert len(graphs) > 0, "Graph list cannot be empty"

    tracemalloc.start()

    def objective(trial: Any) -> float:
        tracemalloc.reset_peak()
        if graph_model == "graph2vec":
            wl_iterations = trial.suggest_categorical(
                "wl_iterations", [x for x in range(2, 5)]
            )
            learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-1, log=True)

            model_g2v = Graph2Vec(
                wl_iterations=wl_iterations,
                dimensions=out_channels,
                learning_rate=learning_rate,
            )
            model_g2v.fit(graphs)
            embeddings = model_g2v.get_embeddings()
        elif graph_model == "deepwalk":
            walk_number = trial.suggest_categorical("walk_number", [2, 5, 7])
            walk_length = trial.suggest_categorical("walk_length", [5, 7, 10])
            window_size = trial.suggest_categorical("window_size", [3, 5, 10])
            learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-1, log=True)

            model_dw = DeepWalk(
                walk_number=walk_number,
                walk_length=walk_length,
                dimensions=out_channels,
                window_size=window_size,
                learning_rate=learning_rate,
            )
            model_dw.fit(graphs)
            embeddings = model_dw.get_embeddings()
        else:
            kernel = trial.suggest_categorical("kernel", ["heat", "wave"])

            if kernel == "heat":
                t_min = trial.suggest_float("t_min", 1e-3, 1e-1, log=True)
                t_max = trial.suggest_float("t_max", 10, 100, log=True)
                timescales = np.logspace(np.log10(t_min), np.log10(t_max), out_channels)
            else:
                max_t = trial.suggest_float("max_t", np.pi, 2 * np.pi)
                timescales = np.linspace(0, max_t, out_channels)

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

        X_train, X_test, y_train, y_test = train_test_split(
            embeddings,
            labels,
            test_size=test_size,
            random_state=42,
            stratify=labels,
        )

        metrics = train_complete_classifier(
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            num_classes=num_classes,
            classifier=classifier,
            mode=mode,
            device=device,
            save_model=False,
        )

        _, peak = tracemalloc.get_traced_memory()
        peak_memory_mb = peak / (1024 * 1024)

        checkpoint = {
            "trial_params": trial.params,
            "F1": metrics["F1"],
            "AUROC": metrics["AUROC"],
            "Accuracy": metrics["Accuracy"],
            "peak_memory_mb": peak_memory_mb,
        }
        trial.set_user_attr("checkpoint", checkpoint)

        return float(metrics["F1"])

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=epochs)

    best_trial = study.best_trial
    best_checkpoint = best_trial.user_attrs["checkpoint"]
    best_hyperparams = best_checkpoint["trial_params"]
    best_model: Union[Graph2Vec, NetLSD, DeepWalk]

    if graph_model == "graph2vec":
        best_model_g2v = Graph2Vec(
            wl_iterations=best_hyperparams["wl_iterations"],
            dimensions=out_channels,
            learning_rate=best_hyperparams["learning_rate"],
        )
        best_model_g2v.fit(graphs)
        embeddings = best_model_g2v.get_embeddings()
        best_model = best_model_g2v
    elif graph_model == "deepwalk":
        best_model_dw = DeepWalk(
            walk_number=best_hyperparams["walk_number"],
            walk_length=best_hyperparams["walk_length"],
            dimensions=out_channels,
            window_size=best_hyperparams["window_size"],
            learning_rate=best_hyperparams["learning_rate"],
        )
        best_model_dw.fit(graphs)
        embeddings = best_model_dw.get_embeddings()
        best_model = best_model_dw
    else:
        eigenvalues_choice = best_hyperparams["eigenvalues"]
        eigenvalues_param: Union[str, int, Tuple[int, int]]
        if isinstance(eigenvalues_choice, str) and eigenvalues_choice.startswith("("):
            eigenvalues_param = ast.literal_eval(eigenvalues_choice)
        else:
            eigenvalues_param = eigenvalues_choice

        kernel = best_hyperparams["kernel"]
        if kernel == "heat":
            t_min = best_hyperparams["t_min"]
            t_max = best_hyperparams["t_max"]
            recalculated_timescales = np.logspace(
                np.log10(t_min), np.log10(t_max), out_channels
            )
        else:
            max_t = best_hyperparams["max_t"]
            recalculated_timescales = np.linspace(0, max_t, out_channels)

        best_model_netlsd = NetLSD(
            timescales=recalculated_timescales,
            kernel=kernel,
            eigenvalues=eigenvalues_param,
            normalization=best_hyperparams["normalization"],
            normalized_laplacian=best_hyperparams["normalized_laplacian"],
        )
        embeddings_raw = [best_model_netlsd.fit_transform(g) for g in graphs]
        embeddings = convert_embeddings_to_real(embeddings_raw)
        best_model = best_model_netlsd

    best_model.save(model_name)
    print(f"Model saved at {model_name}")

    X_train, X_test, y_train, y_test = train_test_split(
        embeddings,
        labels,
        test_size=test_size,
        random_state=42,
        stratify=labels,
    )

    metrics = train_complete_classifier(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        num_classes=num_classes,
        classifier=classifier,
        mode=mode,
        device=device,
        save_model=False,
    )

    tracemalloc.stop()
    print(f"Best Hyperparams Peak Memory: {best_checkpoint['peak_memory_mb']:.2f} MB")

    return best_model, metrics
