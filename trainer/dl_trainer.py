from typing import Any, Dict

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
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

from ml_models.classification.mlp import MLP, MLPDataset
from trainer.utils import EarlyStopper, save_model


def train_step(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str = "cuda",
) -> dict:
    model.train()

    stats = {
        "train_acc": 0.0,
        "train_auc": 0.0,
        "train_recall": 0.0,
        "train_precision": 0.0,
        "train_specificity": 0.0,
        "train_f1": 0.0,
    }

    mode = "binary"  # change in the future, if needed

    acc = Accuracy(task=mode).to(device)
    auc = AUROC(task=mode).to(device)
    recall = Recall(task=mode).to(device)
    precision = Precision(task=mode).to(device)
    specificity = Specificity(task=mode).to(device)
    f1_score = F1Score(task=mode).to(device)

    for X, y in dataloader:
        X, y = X.to(device), y.to(device)

        X, y = X.float(), y.float()

        optimizer.zero_grad()

        y_pred = model(X)

        loss = loss_fn(y_pred, y)

        loss.backward()
        optimizer.step()

        y_pred_classes = (torch.sigmoid(y_pred) > 0.5).float()
        acc.update(y_pred_classes, y)
        auc.update(y_pred, y)
        recall.update(y_pred_classes, y)
        precision.update(y_pred_classes, y)
        specificity.update(y_pred_classes, y)
        f1_score.update(y_pred_classes, y)

    stats["train_acc"] = acc.compute().item()
    stats["train_auc"] = auc.compute().item()
    stats["train_recall"] = recall.compute().item()
    stats["train_precision"] = precision.compute().item()
    stats["train_specificity"] = specificity.compute().item()
    stats["train_f1"] = f1_score.compute().item()

    return stats


def test_step(
    model: torch.nn.Module,
    dataloader: DataLoader,
    loss_fn: torch.nn.Module,
    device: str = "cuda",
) -> dict:

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

    mode = "binary"  # change in the future, if needed

    acc = Accuracy(task=mode).to(device)
    auc = AUROC(task=mode).to(device)
    recall = Recall(task=mode).to(device)
    precision = Precision(task=mode).to(device)
    specificity = Specificity(task=mode).to(device)
    f1_score = F1Score(task=mode).to(device)
    confmat = ConfusionMatrix(task=mode).to(device)

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            X, y = X.float(), y.float()

            test_pred = model(X)
            _ = loss_fn(test_pred, y)

            y_pred_classes = (torch.sigmoid(test_pred) > 0.5).float()
            acc.update(y_pred_classes, y)
            auc.update(test_pred, y)
            recall.update(y_pred_classes, y)
            precision.update(y_pred_classes, y)
            specificity.update(y_pred_classes, y)
            f1_score.update(y_pred_classes, y)
            confmat.update(y_pred_classes, y)
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
    train_dataloader: DataLoader,
    test_dataloader: DataLoader,
    loss_fn: torch.nn.Module,
    optimizer: torch.optim,
    batch_size: int = 128,
    epochs: int = 100,
    patience: int = 10,
    device: str = "cuda",
    target_dir: str = ".",
    model_name: str = "best_MLP.pth",
) -> dict:
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
    }
    best_res = 0.0

    early_stopper = EarlyStopper(patience=patience, increase=True)

    for epoch in tqdm(range(epochs)):
        train_stats = train_step(
            model=model,
            dataloader=train_dataloader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=device,
        )

        test_stats = test_step(
            model=model,
            dataloader=test_dataloader,
            loss_fn=loss_fn,
            device=device,
        )

        if test_stats["test_acc"] > best_res:
            best_res = test_stats["test_acc"]
            save_model(model, target_dir, model_name)
            print(
                f"[INFO] New best model saved at {target_dir} with test accuracy: {best_res:.4f}"
            )

        if early_stopper.early_stop(test_stats["test_acc"]):
            print("Early stopping the training!")
            break

        print(
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

    return results


if __name__ == "__main__":
    df = pd.read_csv("../../../Downloads/Titanic-Dataset.csv")

    df["Age"].fillna(df["Age"].median(), inplace=True)
    df["Fare"].fillna(df["Fare"].median(), inplace=True)

    df["Sex"] = df["Sex"].map({"male": 0, "female": 1})
    df["Embarked"].fillna("S", inplace=True)
    df["Embarked"] = df["Embarked"].map({"S": 0, "C": 1, "Q": 2})

    X = df[["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]]
    y = df["Survived"].values

    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    X_train = X_train.reset_index(drop=True)
    X_val = X_val.reset_index(drop=True)

    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_val = scaler.transform(X_val)

    train_ds = MLPDataset(X_train, y_train)
    val_ds = MLPDataset(X_val, y_val)

    train_dataloader = DataLoader(train_ds, batch_size=32, shuffle=True)
    test_dataloader = DataLoader(val_ds, batch_size=32, shuffle=True)

    model = MLP(device="mps", init_input=len(X_train[0]))
    loss = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(
        model.parameters(), lr=0.001, weight_decay=0.1, amsgrad=True
    )

    stats = train(
        model,
        train_dataloader,
        test_dataloader,
        loss_fn=loss,
        optimizer=optimizer,
        batch_size=32,
        patience=10,
        epochs=100,
        device="mps",
    )
