from typing import Optional

import torch
from torch import nn
from torch_geometric.nn import GCNConv, global_mean_pool


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
        dropout: float = 0.5,
        num_classes: Optional[int] = None,
    ) -> None:
        super(GCN, self).__init__()

        assert task in ["node_classification", "graph_classification"]

        self.in_channels = in_channels
        self.hid_channels = hid_channels
        self.out_channels = out_channels
        self.dropout = dropout
        self.task = task

        self.gcn1 = GCNConv(
            in_channels=in_channels, out_channels=hid_channels, dropout=dropout
        )
        self.relu1 = nn.ReLU(inplace=True)
        self.gcn2 = GCNConv(
            in_channels=hid_channels, out_channels=hid_channels, dropout=dropout
        )
        self.relu2 = nn.ReLU(inplace=True)
        self.gcn3 = GCNConv(
            in_channels=hid_channels, out_channels=out_channels, dropout=dropout
        )

        if task == "graph_classification":
            assert num_classes is not None
            self.lin = nn.Linear(out_channels, num_classes)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        x = self.gcn1(x, edge_index)
        x = self.relu1(x)
        x = self.gcn2(x, edge_index)
        x = self.relu2(x)
        x = self.gcn3(x, edge_index)

        if self.task == "graph_classification":
            x = global_mean_pool(x, batch)
            x = self.lin(x)

            return x

        return x
