import json
import time
from itertools import chain
from typing import Any, List, Optional

import networkx as nx


def inference(
    model: Any,
    data: List[nx.Graph],
    model_weights: str,
    out_json: str,
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
    """
    model.load(model_weights)

    out_features = []
    time_per_data = []

    for graph in data:
        start_time = time.time()
        pred = model.infer([graph]).tolist()
        end_time = time.time()

        time_per_data.append(end_time - start_time)
        out_features.append(list(chain.from_iterable(pred)))

    inference_res = {
        "out_features": out_features,
        "y_hat": ground_truth_labels if ground_truth_labels is not None else [],
        "time": time_per_data,
    }

    with open(out_json, "w") as f:
        json.dump(inference_res, f, indent=4)

    return out_features