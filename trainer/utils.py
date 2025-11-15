import logging
import os
from pathlib import Path
from typing import Any, List

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from torch import nn
from torchsummary import summary


class EarlyStopper:
    """
    Performs early stopping as PyTorch doesn't have an implemented one
    """

    def __init__(
        self, patience: int = 1, min_delta: float = 0.0, increase: bool = False
    ) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.increase = increase
        self.best_value = float("-inf") if increase else float("inf")

    def early_stop(self, validation_val: float) -> bool:
        if not self.increase:
            if validation_val <= self.best_value:
                self.best_value = validation_val
                self.counter = 0
            else:
                self.counter += 1
        else:
            if validation_val >= self.best_value:
                self.best_value = validation_val
                self.counter = 0
            else:
                self.counter += 1

        return self.counter > self.patience


def save_torch_model(model: torch.nn.Module, target_dir: str, model_name: str) -> None:
    """
    Saves the model to the target directory with the passed model name

    :param model: the input model
    :type model: nn.Module
    :param target_dir: the target directory
    :type target_dir: str
    :param model_name: the name of the model(with .pth suffix)
    :type model_name: str
    """
    target_dir_path = Path(target_dir)
    target_dir_path.mkdir(parents=True, exist_ok=True)

    assert model_name.endswith(".pth") or model_name.endswith(
        ".pt"
    ), "model_name should end with '.pt' or '.pth'"
    model_save_path = target_dir_path / model_name

    print(f"[INFO] Saving model to: {model_save_path}")
    torch.save(obj=model.state_dict(), f=model_save_path)


def load_torch_model(model: torch.nn.Module, model_weights: str, device: str) -> None:
    state_dict = torch.load(model_weights, map_location=torch.device(device))
    missing_keys, expected_keys = model.load_state_dict(state_dict)

    if missing_keys:
        print(f"Warning: Missing keys in model: {missing_keys}")
    if expected_keys:
        print(f"Warning: Unexpected keys in model: {expected_keys}")


def logging_basic_config(
    verbose: int = 1, content_only: bool = False, filename: str = ""
) -> Any:
    """
    Basic logging configuration for error exceptions

    :param verbose: input verbose. Default value = 1
    :type verbose: int
    :param content_only: If set to True it will output only the needed content. Default value = False
    :type content_only: bool
    :param filename: input filename. Default value = ''
    :type filename: str

    """
    logging_level = {
        0: logging.WARNING,
        1: logging.INFO,
        2: logging.DEBUG,
        3: logging.ERROR,
        4: logging.CRITICAL,
    }
    fmt = (
        " %(message)s" if content_only else "%(levelname)s (%(funcName)s): %(message)s"
    )
    if filename != "" and filename is not None:
        if not os.path.exists(filename):
            dirname, _ = os.path.split(filename)
            if dirname != "":
                os.mkdir(dirname)
        logging.basicConfig(
            level=logging_level[verbose], format=fmt, force=True, filename=filename
        )
    else:
        logging.basicConfig(level=logging_level[verbose], format=fmt, force=True)
    return logging.getLogger()


def compute_metrics(
    y_preds: list,
    y_hat: list,
    mode: str,
    y_preds_proba: list = [],
) -> dict:
    """
    Return needed metrics for classification

    :param y_pred: The model predictions
    :type y_pred: list
    :param y_hat: Ground truth
    :type y_hat: list
    :param y_preds_proba: Only for SVM models for multiclassification
    :type y_preds_proba: list(default=[])

    :return: dictionary with F1, Accuracy and AUC scores
    """
    y_hat_np = np.array(y_hat)
    y_preds_np = np.array(y_preds)
    y_preds_proba_np = np.array(y_preds_proba)
    auc = 0.0

    if y_preds_proba_np.size > 0:
        if mode == "binary":
            auc = roc_auc_score(y_hat_np, y_preds_proba_np[:, 1])
        else:
            auc = roc_auc_score(y_hat_np, y_preds_proba_np, multi_class="ovr")
    else:
        pass

    return {
        "F1": f1_score(y_hat_np, y_preds_np, average="macro"),
        "Accuracy": accuracy_score(y_hat_np, y_preds_np),
        "AUC": auc,
    }


def convert_embeddings_to_real(embeddings_raw: List[np.ndarray]) -> List[np.ndarray]:
    """
    Converts a list of embeddings to real-valued arrays.

    If an embedding is complex (from NetLSD 'wave' kernel), it concatenates
    its real and imaginary parts. Otherwise, it returns the embedding as is

    :param embeddings_raw: The list of raw embeddings.
    :type embeddings_raw: List[np.ndarray]
    :return: A list of real-valued embeddings.
    :rtype: List[np.ndarray]
    """
    processed_embeddings = []
    for emb in embeddings_raw:
        if np.iscomplexobj(emb):
            processed_embeddings.append(np.concatenate((emb.real, emb.imag)))
        else:
            processed_embeddings.append(emb)

    return processed_embeddings


def get_model_summary(model: nn.Module, in_dim: tuple) -> None:
    """
    Returns summary of the model(total parameters, trainable parameters, etc.)

    :param model: The input model
    :type model: nn.Module
    :param in_dim: The input feature dimensions
    :type in_dim: tuple
    """

    print(summary(model, in_dim))
