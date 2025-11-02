import itertools
import time
from typing import Any, Dict, List, Tuple

import networkx as nx
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.svm import SVC

from ml_models.graph_models.graph2vec import Graph2Vec


def grid_search_graph2vec(
    graphs: List[nx.Graph],
    labels: List[int],
    param_grid: Dict[str, List[Any]],
    test_size: float = 0.2,
    cv: int = 5,
    random_state: int = 42,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Perform grid search for Graph2Vec hyperparameters.

    :param graphs: List of NetworkX graphs
    :type graphs: List[nx.Graph]
    :param labels: Graph labels
    :type labels: List[int]
    :param param_grid: Dictionary of parameters to search
    :type param_grid: Dict[str, List[Any]]
    :param test_size: Test set size
    :type test_size: float
    :param cv: Number of cross-validation folds
    :type cv: int
    :param random_state: Random seed
    :type random_state: int

    :returns: Best parameters and all results
    :rtype: Tuple[Dict[str, Any], List[Dict[str, Any]]]

    Example
    _______

    from trainer.graph2vec_trainer import grid_search_graph2vec
    from dataloader.dataloader import ds_to_graphs

    data = ds_to_graphs("data/MUTAG")

    param_grid = {
        "dimensions": [64, 128, 256],
        "wl_iterations": [2, 3],
        "epochs": [50, 100],
    }

    best_params, results = grid_search_graph2vec(
        data["graphs"],
        data["graph_classes"],
        param_grid
    )

    print(f"Best parameters: {best_params}")
    """
    assert len(graphs) == len(labels), "Number of graphs must match number of labels"
    assert len(graphs) > 0, "Graph list cannot be empty"

    print("[INFO] Starting Grid Search for Graph2Vec")
    print(f"[INFO] Number of graphs: {len(graphs)}")
    print(f"[INFO] Number of classes: {len(set(labels))}")
    print(f"[INFO] Test size: {test_size}")
    print(f"[INFO] Cross-validation folds: {cv}")
    print("=" * 80)

    param_combinations = _generate_param_combinations(param_grid)

    print(f"[INFO] Total combinations to test: {len(param_combinations)}")
    print("=" * 80)

    results: List[Dict[str, Any]] = []
    best_score: float = 0.0
    best_params: Dict[str, Any] = param_combinations[0]

    for idx, params in enumerate(param_combinations):
        print(f"\n[{idx + 1}/{len(param_combinations)}] Testing: {params}")

        start_time = time.time()

        model = Graph2Vec(**params)
        model.fit(graphs)
        embeddings = model.get_embeddings()

        training_time = time.time() - start_time

        X_train, X_test, y_train, y_test = train_test_split(
            embeddings, labels, test_size=test_size, random_state=random_state
        )

        # TODO: This shouldn't always be an SVC(classification), for now it's ok,
        # but we should have an {ml_model} as a parameter that will be used here.
        clf = SVC(kernel="rbf", random_state=random_state)
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)
        test_accuracy = accuracy_score(y_test, y_pred)
        test_f1 = f1_score(y_test, y_pred, average="macro")

        cv_scores = cross_val_score(clf, embeddings, labels, cv=cv)
        cv_mean: float = float(cv_scores.mean())
        cv_std: float = float(cv_scores.std())

        # TODO: Should add more metrics. Keep the best F1 score(not CV mean),
        # never seen that before.
        result: Dict[str, Any] = {
            "params": params,
            "training_time": training_time,
            "test_accuracy": test_accuracy,
            "test_f1": test_f1,
            "cv_mean": cv_mean,
            "cv_std": cv_std,
        }
        results.append(result)

        print(f"  Training time: {training_time:.2f}s")
        print(f"  Test accuracy: {test_accuracy:.4f}")
        print(f"  Test F1: {test_f1:.4f}")
        print(f"  CV mean: {cv_mean:.4f} (+/- {cv_std:.4f})")

        if cv_mean > best_score:
            best_score = cv_mean
            best_params = params
            print("  *** New best score! ***")

    print("\n" + "=" * 80)
    print("[INFO] Grid Search Completed")
    print(f"[INFO] Best CV Score: {best_score:.4f}")
    print(f"[INFO] Best Parameters: {best_params}")
    print("=" * 80)

    return best_params, results


def _generate_param_combinations(
    param_grid: Dict[str, List[Any]],
) -> List[Dict[str, Any]]:
    """
    Generate all combinations of parameters.

    :param param_grid: Dictionary of parameters
    :type param_grid: Dict[str, List[Any]]
    :returns: List of parameter combinations
    :rtype: List[Dict[str, Any]]
    """
    keys = list(param_grid.keys())
    values = list(param_grid.values())

    combinations: List[Dict[str, Any]] = []
    for combination in itertools.product(*values):
        combinations.append(dict(zip(keys, combination)))

    return combinations


def train_best_model(
    graphs: List[nx.Graph],
    labels: List[int],
    best_params: Dict[str, Any],
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[Graph2Vec, SVC, Dict[str, float]]:
    """
    Train final model with best parameters.

    :param graphs: List of NetworkX graphs
    :type graphs: List[nx.Graph]
    :param labels: Graph labels
    :type labels: List[int]
    :param best_params: Best parameters from grid search
    :type best_params: Dict[str, Any]
    :param test_size: Test set size
    :type test_size: float
    :param random_state: Random seed
    :type random_state: int

    :returns: Trained Graph2Vec model, classifier, and metrics
    :rtype: Tuple[Graph2Vec, SVC, Dict[str, float]]

    Example
    _______

    model, clf, metrics = train_best_model(
        graphs, labels, best_params
    )
    print(f"Final accuracy: {metrics['accuracy']:.4f}")
    """
    assert len(graphs) == len(labels), "Number of graphs must match number of labels"
    assert len(graphs) > 0, "Graph list cannot be empty"

    print("[INFO] Training final model with best parameters")
    print(f"[INFO] Parameters: {best_params}")

    model = Graph2Vec(**best_params)
    model.fit(graphs)
    embeddings = model.get_embeddings()

    X_train, X_test, y_train, y_test = train_test_split(
        embeddings, labels, test_size=test_size, random_state=random_state
    )

    # TODO: The same as above.
    clf = SVC(kernel="rbf", random_state=random_state)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    accuracy: float = float(accuracy_score(y_test, y_pred))
    f1: float = float(f1_score(y_test, y_pred, average="macro"))

    # TODO: The same as above
    metrics: Dict[str, float] = {
        "accuracy": accuracy,
        "f1": f1,
    }

    print(f"[INFO] Final Test Accuracy: {accuracy:.4f}")
    print(f"[INFO] Final Test F1: {f1:.4f}")

    return model, clf, metrics
