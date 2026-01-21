from typing import Any, Optional

import optuna
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ml_models.classification.svm import SVMModel
from trainer.utils import compute_metrics

LOGGING_FILENAME = "svc_trainer.log"


def train(
    graph_embeddings: list,
    labels: list,
    mode: str,
    classifier_name: Optional[str] = None,
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
        stratify=labels,
        random_state=42,
    )

    def objective(trial: Any) -> float:
        kernel = trial.suggest_categorical("kernel", ["rbf", "linear"])
        svc_c = trial.suggest_float("C", 1e-10, 10, log=True)
        model = SVMModel(
            mode=mode,
            C=svc_c,
            kernel=kernel,
        )
        auc_scorer = "roc_auc" if mode == "binary" else "roc_auc_ovr"
        scoring = ["f1_macro", auc_scorer, "accuracy"]

        pipeline = Pipeline([("scaler", StandardScaler()), ("svm", model.get_model())])

        scores = cross_validate(
            pipeline,
            X_train,
            y_train,
            scoring=scoring,
            cv=5,
        )

        checkpoint = {
            "trial_params": trial.params,
            "f1_score": scores["test_f1_macro"].mean(),
            "AUC": scores[f"test_{auc_scorer}"].mean(),
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
        mode=mode,
        C=best_hyperparams["C"],
        kernel=best_hyperparams["kernel"],
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    best_model.fit(X_train_scaled, y_train)
    y_preds = best_model.predict(X_val_scaled)
    y_preds_proba = best_model.predict_proba(X_val_scaled)
    stats = compute_metrics(
        y_preds=y_preds, y_hat=y_val, mode=mode, y_preds_proba=y_preds_proba
    )

    if classifier_name is not None:
        best_model.save(classifier_name)

    return best_model, stats
