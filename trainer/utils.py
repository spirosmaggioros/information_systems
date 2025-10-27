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
