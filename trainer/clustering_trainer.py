from typing import Any, Dict, List

import numpy as np
import optuna
from sklearn.metrics import adjusted_rand_score
from sklearn.model_selection import KFold
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
    Trains a clustering model using Optuna for hyperparameter optimization.

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
        
        model_params = {
            "n_clusters": num_classes,
            "random_state": 42
        }

        if model_type == "kmeans":
            model_params["max_iter"] = trial.suggest_categorical(
                "max_iter", [100, 300, 500]
            )
            model_params["tol"] = trial.suggest_float(
                "tol", 1e-5, 1e-3, log=True
            )
            ModelClass = KMeansModel
            
        elif model_type == "spectral":
            model_params["affinity"] = trial.suggest_categorical(
                "affinity", ["rbf", "nearest_neighbors"]
            )
            if model_params["affinity"] == "rbf":
                model_params["gamma"] = trial.suggest_float(
                    "gamma", 1.0, 1e2, log=True 
                )
            else:
                model_params["n_neighbors"] = trial.suggest_int(
                    "n_neighbors", 3, 10,
                )
            ModelClass = SpectralClusteringModel
        else:
            raise ValueError("Unsupported model_type")

        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        scores = []
        
        for train_index, test_index in kf.split(X_scaled):
            X_train_fold, X_test_fold = X_scaled[train_index], X_scaled[test_index]
            y_test_fold = y[test_index]

            model = ModelClass(**model_params)
            
            if model_type == 'kmeans':
                model.fit(X_train_fold)
                pred_labels = model.predict(X_test_fold)
            else:
                pred_labels = model.predict(X_test_fold)
            
            ari = adjusted_rand_score(y_test_fold, pred_labels)
            scores.append(ari)

        mean_ari = float(np.mean(scores))
        trial.set_user_attr("checkpoint", {"mean_ARI": mean_ari})
        return mean_ari

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=50)

    best_trial = study.best_trial
    best_hyperparams = best_trial.params
    best_ari = best_trial.value
    
    print(f"Best {model_type} trial: ARI={best_ari:.4f} with params={best_hyperparams}")

    final_params = {
        "n_clusters": num_classes,
        "random_state": 42,
        **best_hyperparams,
    }

    if model_type == "kmeans":
        best_model = KMeansModel(**final_params)
    else:
        best_model = SpectralClusteringModel(**final_params)

    best_model.fit(X_scaled)
    
    if model_type == 'kmeans':
        final_labels = best_model.labels_
    else:
        final_labels = best_model.predict(X_scaled) 

    final_ari = adjusted_rand_score(y, final_labels)

    stats = {
        "Best_CV_ARI": best_ari,
        "Final_Full_Data_ARI": final_ari,
        "Best_Params": best_hyperparams,
        "Final_Pred_Labels": final_labels
    }

    return best_model, stats
