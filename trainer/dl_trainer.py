import time
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchmetrics import (
    AUROC,
    Accuracy,
    ConfusionMatrix,
    F1Score,
    Precision,
    Recall,
    Specificity,
)
from tqdm import tqdm

from trainer.utils import EarlyStopper, logging_basic_config, save_torch_model


def train_step(
    model: nn.Module,
    model_type: str,
    dataloader: DataLoader,
    loss_fn: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    mode: str,
    num_classes: Optional[int] = None,
    device: str = "cuda",
) -> dict:
    assert mode in ["binary", "multiclass"]
    model.train()

    stats = {
        "train_acc": 0.0,
        "train_auc": 0.0,
        "train_recall": 0.0,
        "train_precision": 0.0,
        "train_specificity": 0.0,
        "train_f1": 0.0,
    }

    if mode == "multiclass":
        assert num_classes is not None
        acc = Accuracy(task=mode, average="macro", num_classes=num_classes).to(device)
        auc = AUROC(task=mode, average="macro", num_classes=num_classes).to(device)
        recall = Recall(task=mode, average="macro", num_classes=num_classes).to(device)
        precision = Precision(task=mode, average="macro", num_classes=num_classes).to(
            device
        )
        specificity = Specificity(
            task=mode, average="macro", num_classes=num_classes
        ).to(device)
        f1_score = F1Score(task=mode, average="macro", num_classes=num_classes).to(
            device
        )
    else:
        acc = Accuracy(task=mode).to(device)
        auc = AUROC(task=mode).to(device)
        recall = Recall(task=mode).to(device)
        precision = Precision(task=mode).to(device)
        specificity = Specificity(task=mode).to(device)
        f1_score = F1Score(task=mode).to(device)

    if model_type == "torch":
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)

            if mode != "multiclass":
                X, y = X.float(), y.float()
            else:
                X, y = X.float(), y.long()

            optimizer.zero_grad()

            y_pred = model(X)

            loss = loss_fn(y_pred, y)
            loss.backward()

            optimizer.step()

            y_pred_classes = (
                (torch.sigmoid(y_pred) > 0.5).float()
                if mode == "binary"
                else torch.argmax(y_pred, dim=1)
            )

            acc.update(y_pred_classes, y)
            auc.update(y_pred, y)
            recall.update(y_pred_classes, y)
            precision.update(y_pred_classes, y)
            specificity.update(y_pred_classes, y)
            f1_score.update(y_pred_classes, y)
    else:
        for batch in dataloader:
            batch = batch.to(device)

            optimizer.zero_grad()

            y_pred = model(
                x=batch.node_attributes, edge_index=batch.edge_index, batch=batch.batch
            )
            if isinstance(y_pred, tuple):
                _, y_pred_val = y_pred
            else:
                y_pred_val = y_pred

            loss = loss_fn(y_pred_val, batch.y)

            loss.backward()

            optimizer.step()

            y_pred_classes = (
                (torch.sigmoid(y_pred_val) > 0.5).float()
                if mode == "binary"
                else torch.argmax(y_pred_val, dim=1)
            )

            acc.update(y_pred_classes, batch.y)
            auc.update(y_pred_val, batch.y)
            recall.update(y_pred_classes, batch.y)
            precision.update(y_pred_classes, batch.y)
            specificity.update(y_pred_classes, batch.y)
            f1_score.update(y_pred_classes, batch.y)

    stats["train_acc"] = acc.compute().item()
    stats["train_auc"] = auc.compute().item()
    stats["train_recall"] = recall.compute().item()
    stats["train_precision"] = precision.compute().item()
    stats["train_specificity"] = specificity.compute().item()
    stats["train_f1"] = f1_score.compute().item()

    return stats


def test_step(
    model: torch.nn.Module,
    model_type: str,
    dataloader: DataLoader,
    mode: str,
    num_classes: Optional[int] = None,
    device: str = "cuda",
) -> dict:
    assert mode in ["binary", "multiclass"]
    model.eval()

    stats = {
        "test_acc": 0.0,
        "test_auc": 0.0,
        "test_recall": 0.0,
        "test_precision": 0.0,
        "test_specificity": 0.0,
        "test_f1": 0.0,
        "test_confussion_matrix": [],
    }

    if mode == "multiclass":
        assert num_classes is not None
        acc = Accuracy(task=mode, average="macro", num_classes=num_classes).to(device)
        auc = AUROC(task=mode, average="macro", num_classes=num_classes).to(device)
        recall = Recall(task=mode, average="macro", num_classes=num_classes).to(device)
        precision = Precision(task=mode, average="macro", num_classes=num_classes).to(
            device
        )
        specificity = Specificity(
            task=mode, average="macro", num_classes=num_classes
        ).to(device)
        f1_score = F1Score(task=mode, average="macro", num_classes=num_classes).to(
            device
        )
        confmat = ConfusionMatrix(task=mode, num_classes=num_classes).to(device)
    else:
        acc = Accuracy(task=mode).to(device)
        auc = AUROC(task=mode).to(device)
        recall = Recall(task=mode).to(device)
        precision = Precision(task=mode).to(device)
        specificity = Specificity(task=mode).to(device)
        f1_score = F1Score(task=mode).to(device)
        confmat = ConfusionMatrix(task=mode).to(device)

    with torch.no_grad():
        if model_type == "torch":
            for X, y in dataloader:
                X, y = X.to(device), y.to(device)

                if mode != "multiclass":
                    X, y = X.float(), y.float()
                else:
                    X, y = X.float(), y.long()

                test_pred = model(X)

                y_pred_classes = (
                    (torch.sigmoid(test_pred) > 0.5).float()
                    if mode == "binary"
                    else torch.argmax(test_pred, dim=1)
                )

                acc.update(y_pred_classes, y)
                auc.update(test_pred, y)
                recall.update(y_pred_classes, y)
                precision.update(y_pred_classes, y)
                specificity.update(y_pred_classes, y)
                f1_score.update(y_pred_classes, y)
                confmat.update(y_pred_classes, y)
        else:
            for batch in dataloader:
                batch = batch.to(device)

                test_pred = model(
                    x=batch.node_attributes,
                    edge_index=batch.edge_index,
                    batch=batch.batch,
                )

                if isinstance(test_pred, tuple):
                    _, test_pred_val = test_pred
                else:
                    test_pred_val = test_pred

                y_pred_classes = (
                    (torch.sigmoid(test_pred_val) > 0.5).float()
                    if mode == "binary"
                    else torch.argmax(test_pred_val, dim=1)
                )

                acc.update(y_pred_classes, batch.y)
                auc.update(test_pred_val, batch.y)
                recall.update(y_pred_classes, batch.y)
                precision.update(y_pred_classes, batch.y)
                specificity.update(y_pred_classes, batch.y)
                f1_score.update(y_pred_classes, batch.y)
                confmat.update(y_pred_classes, batch.y)

    stats["test_acc"] = acc.compute().item()
    stats["test_auc"] = auc.compute().item()
    stats["test_recall"] = recall.compute().item()
    stats["test_precision"] = precision.compute().item()
    stats["test_specificity"] = specificity.compute().item()
    stats["test_f1"] = f1_score.compute().item()
    stats["test_confussion_matrix"] = confmat.compute().cpu().numpy()

    return stats


def train(
    model: nn.Module,
    model_type: str,
    train_dataloader: DataLoader,
    test_dataloader: DataLoader,
    loss_fn: torch.nn.Module,
    optimizer: torch.optim,
    mode: str,
    num_classes: Optional[int] = None,
    epochs: int = 200,
    patience: int = 10,
    device: str = "cuda",
    target_dir: str = ".",
    model_name: str = "dl_trainer_best_model.pth",
    save_model: bool = False,
) -> tuple[dict, dict]:
    assert model_type in ["torch", "torch_geometric"]
    logger = logging_basic_config(
        verbose=1, content_only=True, filename="dl_trainer.log"
    )

    results: Dict[str, Any] = {
        "train_acc": [],
        "train_auc": [],
        "train_recall": [],
        "train_precision": [],
        "train_specificity": [],
        "train_f1": [],
        "test_acc": [],
        "test_auc": [],
        "test_recall": [],
        "test_precision": [],
        "test_specificity": [],
        "test_f1": [],
        "test_confussion_matrix": [],
        "eval_acc": 0.0,
        "eval_auc": 0.0,
        "eval_recall": 0.0,
        "eval_precision": 0.0,
        "eval_specificity": 0.0,
        "eval_f1": 0.0,
        "training_time": 0.0,
    }
    best_checkpoint: Dict[str, Any] = {
        "test_acc": [],
        "test_auc": [],
        "test_recall": [],
        "test_precision": [],
        "test_specificity": [],
        "test_f1": [],
        "test_confussion_matrix": [],
    }

    best_res = 0.0

    early_stopper = EarlyStopper(patience=patience, increase=True)
    start_training_time = time.time()

    for epoch in tqdm(range(epochs)):
        train_stats = train_step(
            model=model,
            model_type=model_type,
            dataloader=train_dataloader,
            num_classes=num_classes,
            loss_fn=loss_fn,
            optimizer=optimizer,
            mode=mode,
            device=device,
        )

        test_stats = test_step(
            model=model,
            model_type=model_type,
            dataloader=test_dataloader,
            num_classes=num_classes,
            mode=mode,
            device=device,
        )

        if test_stats["test_f1"] > best_res:
            best_res = test_stats["test_f1"]
            if save_model:
                save_torch_model(model, target_dir, model_name)
                logger.info(
                    f"New best model saved at {target_dir} with test F1: {best_res:.4f}"
                )
            best_checkpoint["test_acc"] = test_stats["test_acc"]
            best_checkpoint["test_auc"] = test_stats["test_auc"]
            best_checkpoint["test_recall"] = test_stats["test_recall"]
            best_checkpoint["test_precision"] = test_stats["test_precision"]
            best_checkpoint["test_specificity"] = test_stats["test_specificity"]
            best_checkpoint["test_f1"] = test_stats["test_f1"]
            best_checkpoint["test_confussion_matrix"] = test_stats[
                "test_confussion_matrix"
            ]

        if early_stopper.early_stop(test_stats["test_acc"]):
            logger.info("Early stopping the training!")
            break

        logger.info(
            f"Epoch: {epoch+1} | "
            f"train_acc: {train_stats['train_acc']:.4f} | "
            f"train_auc: {train_stats['train_auc']:.4f} | "
            f"train_recall: {train_stats['train_recall']:.4f} | "
            f"train_precision: {train_stats['train_precision']:.4f} | "
            f"train_specificity: {train_stats['train_specificity']:.4f} | "
            f"train_f1: {train_stats['train_f1']:.4f} | "
            f"test_acc: {test_stats['test_acc']:.4f} | "
            f"test_auc: {test_stats['test_auc']:.4f} | "
            f"test_recall: {test_stats['test_recall']:.4f} | "
            f"test_precision: {test_stats['test_precision']:.4f} | "
            f"test_specificity: {test_stats['test_specificity']:.4f} | "
            f"test_f1: {test_stats['test_f1']:.4f} | "
            f"test_confussion_matrix: {test_stats['test_confussion_matrix']}"
        )

        results["train_acc"].append(train_stats["train_acc"])
        results["train_auc"].append(train_stats["train_auc"])
        results["train_recall"].append(train_stats["train_recall"])
        results["train_precision"].append(train_stats["train_precision"])
        results["train_specificity"].append(train_stats["train_specificity"])
        results["train_f1"].append(train_stats["train_f1"])
        results["test_acc"].append(test_stats["test_acc"])
        results["test_auc"].append(test_stats["test_auc"])
        results["test_recall"].append(test_stats["test_recall"])
        results["test_precision"].append(test_stats["test_precision"])
        results["test_specificity"].append(test_stats["test_specificity"])
        results["test_f1"].append(test_stats["test_f1"])

    results["training_time"] = time.time() - start_training_time
    logger.info(f"[INFO] Training finished at {results['training_time']}")

    return results, best_checkpoint
