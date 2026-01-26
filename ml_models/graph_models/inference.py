import json
import time
import tracemalloc
from itertools import chain
from typing import Any, List, Optional

import networkx as nx
import numpy as np
from sklearn.preprocessing import StandardScaler

from ml_models.classification.mlp import MLP
from ml_models.classification.svm import SVMModel


def inference(
    model: Any,
    data: List[nx.Graph],
    model_weights: str,
    out_json: str,
    classifier: Optional[Any] = None,
    classifier_weights: Optional[str] = None,
    ground_truth_labels: Optional[list] = None,
) -> list:
    """
    Load weights and perform inference on passed data, saving results to passed json file

    :param model: The passed graph model
    :type model: Union[Graph2Vec, netLSD, DeepWalk]
    :param data: Graphs to perform inference on
    :type data: List[nx.Graph]
    :param model_weights: The complete path to the model weights
    :type model_weights: str
    :param out_json: The complete path(with .json suffix) to save predictions
    :type out_json: str
    :param classifier: Either an SVM or an MLP model
    :type classifier: Any(SVMModel or MLP class)
    :param classifier_weights: absolute path to classifier's weights
    :type classifier_weights: Optional[str](default=None)
    """
    tracemalloc.start()
    if classifier is not None:
        assert classifier_weights is not None
        if isinstance(classifier, MLP):
            print("Inference not yet implemented for MLP!")
            exit(0)
        else:
            assert isinstance(classifier, SVMModel)
            classifier.load(classifier_weights)

    model.load(model_weights)

    out_features = []
    time_per_data = []

    for graph in data:
        start_time = time.time()

        raw_pred = model.infer([graph])

        if np.iscomplexobj(raw_pred):
            raw_pred = np.concatenate((raw_pred.real, raw_pred.imag), axis=1)

        pred = raw_pred.tolist()

        end_time = time.time()

        time_per_data.append(end_time - start_time)
        out_features.append(list(chain.from_iterable(pred)))

    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    peak_memory_mb = peak / (1024 * 1024)

    scaled_features = StandardScaler().fit_transform(out_features)

    inference_res = {
        "out_features": out_features,
        "y_hat": ground_truth_labels if ground_truth_labels is not None else [],
        "time": time_per_data,
        "peak_memory_mb": peak_memory_mb,
    }

    if classifier is not None:
        y_preds = list(classifier.predict(scaled_features))
        inference_res["y_preds"] = [int(x) for x in y_preds]

    with open(out_json, "w") as f:
        json.dump(inference_res, f, indent=4)

    return out_features
