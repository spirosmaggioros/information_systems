from typing import Optional, Tuple

import torch
from torch import nn
from torch_geometric.nn import global_mean_pool
from torch_geometric.nn.models import PNA

from ml_models.utils import init_weights


class _PNA(nn.Module):
    """
    Helper class for the PNA model

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
        deg_hist: torch.Tensor,
        num_layers: int = 4,
        dropout: float = 0.0,
        num_classes: Optional[int] = None,
    ) -> None:
        super(_PNA, self).__init__()

        assert task in ["node_classification", "graph_classification"]

        self.task = task
        self.pna = PNA(
            in_channels=in_channels,
            hidden_channels=hid_channels,
            out_channels=out_channels,
            num_layers=num_layers,
            dropout=dropout,
            aggregators=["sum", "mean", "min", "max"],
            scalers=["identity", "amplification", "attenuation"],
            deg=deg_hist,
        )

        if task == "graph_classification":
            assert num_classes is not None
            self.lin = nn.Linear(out_channels, num_classes)

        init_weights(self.pna)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        x = self.pna(x, edge_index, batch)

        if self.task == "graph_classification":
            x = global_mean_pool(x, batch)
            return x, self.lin(x)

        return x
