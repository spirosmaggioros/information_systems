import json
import time
from typing import Optional, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader

from trainer.utils import load_torch_model


def inference(
    model: nn.Module,
    mode: str,
    dataloader: DataLoader,
    model_weights: str,
    out_json: str = "",
    ground_truth_labels: Optional[list] = None,
    device: str = "cpu",
) -> Tuple[list, list]:
    """
    Load weights and perform inference on passed dataloader, saving results to passed json file
    (For torch geometric models only)

    :param model: The passed model
    :type model: nn.Module
    :param mode: Either binary or multiclass
    :type mode: str
    :param dataloader: Input data
    :type dataloader: DataLoader
    :param model_weights: The complete path to the model weights
    :type model_weights: str
    :param out_json: The complete path(with .json suffix) to save predictions
    :type out_json: str
    """
    load_torch_model(model, model_weights, device)
    model.eval()

    y_preds = []
    y_features = []
    time_per_data = []
    mem_allocated = []

    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)

            if device == "cuda":
                torch.cuda.reset_peak_memory_stats(device=None)

            start = time.time()
            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                y_pred = model(
                    x=batch.node_attributes,
                    edge_index=batch.edge_index,
                    batch=batch.batch,
                )
            if isinstance(y_pred, tuple):
                y_pred_features, y_pred_val = y_pred
            else:
                y_pred_val = y_pred

            time_per_data.append(time.time() - start)

            y_pred_classes = (
                (torch.sigmoid(y_pred_val) > 0.5).float()
                if mode == "binary"
                else torch.argmax(y_pred_val, dim=1)
            )

            y_preds.append(y_pred_classes.cpu().flatten().tolist()[0])
            y_features.append(y_pred_features.cpu().flatten().tolist())
            mem_allocated.append(
                torch.cuda.max_memory_allocated(device=None) / (1024**2)
            )

    if out_json != "":
        inference_res = {
            "predictions": y_preds,
            "out_features": y_features,
            "y_hat": ground_truth_labels if ground_truth_labels is not None else [],
            "mem_allocated_per_data": mem_allocated,
            "time": time_per_data,
        }

        with open(out_json, "w") as f:
            json.dump(inference_res, f, indent=4)

    return y_features, y_preds
