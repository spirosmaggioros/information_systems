from typing import Optional, Tuple

import torch
from torch import nn
from torch_geometric.nn import GIN, global_mean_pool

from ml_models.utils import init_weights


class _GIN(nn.Module):
    """
    Helper class for the GIN model

    :param task: Either node_classification or graph_classification
    :type task: str
    :param in_channels: the input channels(features of the data)
    :type in_channels: int
    :param hid_channels: the number of hidden channels for the GAT convolutional layers
    :type hid_channels: int
    :param out_channels: the number of output channels(number of classes)
    :type out_channels: int
    :param num_layers: the number of layers of GINConv
    :type num_layers: int
    :param dropout: the dropout value for the GINConv layers
    :type dropout: float
    :param num_classes: set only for graph classification
    :type num_classes: Optional[int](default = None)

    Example
    _______
    device = torch.device('cuda')
    model = _GIN(task="graph_classification", in_channels=18, hid_channels=32, out_channels=64, num_classes=6)
    """

    def __init__(
        self,
        task: str,
        in_channels: int,
        hid_channels: int,
        out_channels: int,
        num_layers: int = 4,
        dropout: float = 0.0,
        num_classes: Optional[int] = None,
    ) -> None:
        super(_GIN, self).__init__()

        assert task in ["node_classification", "graph_classification"]

        self.task = task

        self.gin = GIN(
            in_channels=in_channels,
            hidden_channels=hid_channels,
            out_channels=out_channels,
            num_layers=num_layers,
            dropout=dropout,
        )

        if task == "graph_classification":
            assert num_classes is not None
            self.lin = nn.Linear(out_channels, num_classes)

        self.gin.apply(init_weights)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        x = self.gin(x, edge_index, batch)

        if self.task == "graph_classification":
            x = global_mean_pool(x, batch)
            return x, self.lin(x)

        return x
