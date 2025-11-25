from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import ModuleList
from torch_geometric.nn import BatchNorm, GCNConv, global_mean_pool

from ml_models.utils import init_weights


class GCN(nn.Module):
    """
    Implementation of a Graph Convolutional Network

    :param task: Either node_classification or graph_classification
    :type task: str
    :param in_channels: the input channels(features of the data)
    :type in_channels: int
    :param hid_channels: the number of hidden channels for the GAT convolutional layers
    :type hid_channels: int
    :param out_channels: the number of output channels(number of classes)
    :type out_channels: int
    :param num_layers: the total number of convolutional layers of the model(default = 4)
    :type num_layers: int
    :param dropout: the dropout value for GCNConv layers
    :type dropout: float
    :param num_classes: Optional, only when task is graph classification
    :type num_classes: Optional[int](default=None)

    Example
    _______
    device = torch.device('cuda')
    model = GCN(task="node_classification", in_channels=1433, hid_channels=8, out_channels=7).to(device)
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
        super(GCN, self).__init__()

        assert task in ["node_classification", "graph_classification"]

        self.in_channels = in_channels
        self.hid_channels = hid_channels
        self.out_channels = out_channels
        self.dropout = dropout
        self.task = task

        self.convs = ModuleList()
        self.batch_norms = ModuleList()
        self.dropouts = ModuleList()
        for _ in range(num_layers):
            self.convs.append(
                GCNConv(
                    in_channels=in_channels,
                    out_channels=hid_channels,
                )
            )
            self.batch_norms.append(BatchNorm(hid_channels))
            self.dropouts.append(nn.Dropout(dropout))

        if task == "graph_classification":
            assert num_classes is not None
            self.lin = nn.Linear(out_channels, num_classes)

        self.apply(init_weights)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:

        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.batch_norms):
                x = self.batch_norms[i](x)
            x = self.dropouts[i](x)
            x = F.relu(x)

        if self.task == "graph_classification":
            x = global_mean_pool(x, batch)
            return x, self.lin(x)

        return x
