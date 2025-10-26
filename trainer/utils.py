from pathlib import Path

import torch


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


def save_model(model: torch.nn.Module, target_dir: str, model_name: str) -> None:

    target_dir_path = Path(target_dir)
    target_dir_path.mkdir(parents=True, exist_ok=True)

    assert model_name.endswith(".pth") or model_name.endswith(
        ".pt"
    ), "model_name should end with '.pt' or '.pth'"
    model_save_path = target_dir_path / model_name

    print(f"[INFO] Saving model to: {model_save_path}")
    torch.save(obj=model.state_dict(), f=model_save_path)
