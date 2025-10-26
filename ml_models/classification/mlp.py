from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

from ml_models.utils import init_weights


class MLPDataset(Dataset):
    """
    A class for managing datasets that will be used for the MLP trainingA

    :param X: the input data
    :type X: list
    :param y: the labels
    :type y: list
    """

    def __init__(self, X: list, y: list) -> None:
        self.X = np.array(X, dtype=np.float32)
        self.y = np.array(y, dtype=np.float32)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return torch.tensor(self.X[idx], dtype=torch.float32), torch.tensor(
            self.y[idx],
            dtype=torch.float32,
        )


class MLP(nn.Module):
    def __init__(
        self,
        hidden_size: int = 256,
        task: str = "classification",
        dropout: float = 0.2,
        use_bn: bool = False,
        bn: str = "bn",
        device: str = "cuda",
        init_input: int = -1,
    ) -> None:
        """
        Simple MLP model implemented in torch

        :param hidden_size: the amount of hidden features
        :type hidden_size: int
        :param task: either classification or regression
        :type task: str
        :param dropout: dropout value
        :type dropout: float
        """
        super(MLP, self).__init__()

        self.hidden_size = hidden_size
        self.classification = True if task == "classification" else False
        self.dropout = dropout
        self.use_bn = use_bn
        self.bn = bn

        def mlp_layer(hidden_size: int) -> nn.Module:
            layers = [nn.LazyLinear(hidden_size)]
            if self.use_bn:
                if self.bn == "bn":
                    layers.append(nn.BatchNorm1d(hidden_size, eps=1e-5))
                else:
                    layers.append(nn.LayerNorm(hidden_size, eps=1e-5))

            layers += [nn.ReLU(), nn.Dropout(p=self.dropout)]
            return nn.Sequential(*layers)

        self.model = nn.Sequential(
            mlp_layer(self.hidden_size),
            mlp_layer(self.hidden_size // 2),
            nn.LazyLinear(1),
        ).to(device)

        if init_input > 0:
            init_input = torch.randn(1, 1, init_input).to(device)
            self.model(init_input)

            init_weights(self.model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x).squeeze()
