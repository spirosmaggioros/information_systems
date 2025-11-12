import time

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from trainer.utils import load_torch_model


def inference(
    model: nn.Module,
    mode: str,
    dataloader: DataLoader,
    model_weights: str,
    out_csv: str,
    device: str = "cpu",
) -> None:
    """
    Load weights and perform inference on passed dataloader, saving results to passed csv file
    (For torch models only)

    :param model: The passed model
    :type model: nn.Module
    :param mode: Either binary or multiclass
    :type mode: str
    :param dataloader: Input data
    :type dataloader: DataLoader
    :param model_weights: The complete path to the model weights
    :type model_weights: str
    :param out_csv: The complete path(with .csv suffix) to save predictions
    :type out_csv: str
    """
    load_torch_model(model, model_weights, device)

    y_preds = []
    time_per_data = []
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)

            if mode != "multiclass":
                X, y = X.float(), y.float()
            else:
                X, y = X.float(), y.long()

            start = time.time()
            y_pred = model(X)
            time_per_data.append(time.time() - start)

            y_pred_classes = (
                (torch.sigmoid(y_pred) > 0.5).float()
                if mode == "binary"
                else torch.argmax(y_pred, dim=1)
            )

            y_preds.append(y_pred_classes.cpu().tolist())

    inference_res = pd.DataFrame(
        {
            "Preds": y_preds,
            "Time": time_per_data,
        }
    )
    inference_res.to_csv(out_csv, index=False)
