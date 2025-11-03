from typing import Any

import optuna
from sklearn.model_selection import cross_validate, train_test_split

from ml_models.classification.svm import SVMModel
from trainer.utils import compute_metrics

LOGGING_FILENAME = "svc_trainer.log"


def train(
    graph_embeddings: list,
    labels: list,
) -> Any:
    """
    SVC trainer

    :param graph_embeddings: Graph embeddings computed by a graph model
    :type graph_embeddings: list
    :param labels: Labels of graphs/nodes for each graph embedding
    :type labels: list

    :return: The best estima"tor of the optuna study
    """

    X_train, X_val, y_train, y_val = train_test_split(
        graph_embeddings,
        labels,
        test_size=0.25,
        random_state=42,
    )

    def objective(trial: Any) -> float:
        kernel = trial.suggest_categorical("kernel", ["rbf", "linear"])
        svc_c = trial.suggest_float("C", 1e-10, 1e10, log=True)
        model = SVMModel(
            C=svc_c,
            kernel=kernel,
        )
        scoring = ["f1_macro", "roc_auc_ovr", "accuracy"]
        scores = cross_validate(
            model.get_model(),
            X_train,
            y_train,
            scoring=scoring,
            cv=5,
        )

        checkpoint = {
            "trial_params": trial.params,
            "f1_score": scores["test_f1_macro"].mean(),
            "AUC": scores["test_roc_auc_ovr"].mean(),
            "Accuracy": scores["test_accuracy"].mean(),
        }

        trial.set_user_attr("checkpoint", checkpoint)
        return float(scores["test_f1_macro"].mean())

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=50)

    best_trial = study.best_trial
    best_checkpoint = best_trial.user_attrs["checkpoint"]
    best_hyperparams = best_checkpoint["trial_params"]
    best_model = SVMModel(
        C=best_hyperparams["C"],
        kernel=best_hyperparams["kernel"],
    )
    best_model.fit(X_train, y_train)
    y_preds = best_model.predict(X_val)
    y_preds_proba = best_model.predict_proba(X_val)
    stats = compute_metrics(y_preds, y_val, y_preds_proba)

    return best_model, stats
