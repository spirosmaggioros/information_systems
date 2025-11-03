from typing import Any, Dict, List, Tuple

import networkx as nx
import optuna
import torch
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader

from ml_models.classification.mlp import MLP, MLPDataset
from ml_models.graph_models.graph2vec import Graph2Vec
from trainer.dl_trainer import train as train_dl
from trainer.svm_trainer import train as train_svm


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
        y_train_relabeled = [y - 1 for y in y_train]
        y_test_relabeled = [y - 1 for y in y_test]
        train_ds = MLPDataset(X_train, y_train_relabeled)
        val_ds = MLPDataset(X_test, y_test_relabeled)

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
    graphs: List[nx.Graph],
    labels: List[int],
    num_classes: int,
    mode: str,
    test_size: float = 0.25,
    classifier: str = "SVC",
    device: str = "mps",
) -> Tuple[Graph2Vec, Dict[str, float]]:
    """
    Train a graph2vec model

    :param graphs: List of NetworkX graphs
    :type graphs: List[nx.Graph]
    :param labels: Graph labels
    :type labels: List[int]
    :param test_size: Test set size
    :type test_size: float

    :returns: Trained Graph2Vec model, classifier, and metrics
    :rtype: Tuple[Graph2Vec, Dict[str, float]]

    Example
    _______

    model, metrics = train_best_model(
        graphs=graphs,
        labels=labels,
        num_classes=3,
    )
    print(f"Final accuracy: {metrics['Accuracy']:.4f}")
    print(f"Final F1: {metrics['F1']:.4f}")
    """
    assert len(graphs) == len(labels), "Number of graphs must match number of labels"
    assert len(graphs) > 0, "Graph list cannot be empty"

    def objective(trial: Any) -> float:
        wl_iterations = trial.suggest_categorical(
            "wl_iterations", [x for x in range(2, 4)]
        )
        dimensions = trial.suggest_categorical(
            "dimensions", [x for x in range(64, 256, 32)]
        )
        learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-1, log=True)

        model = Graph2Vec(
            wl_iterations=wl_iterations,
            dimensions=dimensions,
            learning_rate=learning_rate,
        )
        model.fit(graphs)
        embeddings = model.get_embeddings()
        X_train, X_test, y_train, y_test = train_test_split(
            embeddings, labels, test_size=test_size, random_state=42
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
        checkpoint = {
            "trial_params": trial.params,
            "F1": metrics["F1"],
            "AUROC": metrics["AUROC"],
            "Accuracy": metrics["Accuracy"],
        }
        trial.set_user_attr("checkpoint", checkpoint)

        return float(metrics["F1"])

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=100)

    best_trial = study.best_trial
    best_checkpoint = best_trial.user_attrs["checkpoint"]
    best_hyperparams = best_checkpoint["trial_params"]
    best_model = Graph2Vec(
        wl_iterations=best_hyperparams["wl_iterations"],
        dimensions=best_hyperparams["dimensions"],
        learning_rate=best_hyperparams["learning_rate"],
    )
    best_model.fit(graphs)
    embeddings = best_model.get_embeddings()
    X_train, X_test, y_train, y_test = train_test_split(
        embeddings,
        labels,
        test_size=test_size,
        random_state=42,
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
        save_model=True,
    )

    return best_model, metrics
