import json
import time
import tracemalloc
from itertools import chain
from typing import Any, List, Optional

import networkx as nx
import numpy as np  # <--- Essential import

def inference(
    model: Any,
    data: List[nx.Graph],
    model_weights: str,
    out_json: str,
    ground_truth_labels: Optional[list] = None,
) -> list:
    """
    Load weights and perform inference on passed data, saving results to passed json file.
    Tracks time per graph and total peak memory usage.
    Handles complex numbers by concatenating real and imaginary parts.
    """
    # 1. Start Memory Tracking
    tracemalloc.start()

    # 2. Load Model
    model.load(model_weights)

    out_features = []
    time_per_data = []

    # 3. Run Inference Loop
    for graph in data:
        start_time = time.time()
        
        # Get raw prediction (returns a numpy array, usually shape (1, dims))
        raw_pred = model.infer([graph])
        
        # --- YOUR FIX APPLIED HERE ---
        # If the embedding is complex (NetLSD Wave Kernel), concatenate Real + Imag parts
        if np.iscomplexobj(raw_pred):
            raw_pred = np.concatenate((raw_pred.real, raw_pred.imag), axis=1)
        # -----------------------------

        pred = raw_pred.tolist()
        end_time = time.time()

        time_per_data.append(end_time - start_time)
        out_features.append(list(chain.from_iterable(pred)))

    # 4. Stop Memory Tracking & Get Peak
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # Convert Peak to Megabytes (MB)
    peak_memory_mb = peak / (1024 * 1024)

    # 5. Save Results
    inference_res = {
        "out_features": out_features,
        "y_hat": ground_truth_labels if ground_truth_labels is not None else [],
        "time": time_per_data,
        "peak_memory_mb": peak_memory_mb,
    }

    with open(out_json, "w") as f:
        json.dump(inference_res, f, indent=4)

    return out_features