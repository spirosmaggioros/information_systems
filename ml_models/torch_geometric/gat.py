from typing import Optional

import torch
from torch import nn
from torch_geometric.nn import GATv2Conv, global_mean_pool

from ml_models.utils import init_weights


class GAT(nn.Module):
    """
    Implementation of a Graph Attention Network

    :param task: Either node_classification or graph_classification
    :type task: str
    :param in_channels: the input channels(features of the data)
    :type in_channels: int
    :param hid_channels: the number of hidden channels for the GAT convolutional layers
    :type hid_channels: int
    :param out_channels: the number of output channels(number of classes)
    :type out_channels: int
    :param in_head: the number of heads to use for multihead attention
    :type in_head: int
    :param out_head: the number of heads to use in the final layer of GATConv
    :type out_head: int
    :param dropout: the dropout value for GATConv layers
    :type dropout: float
    :param num_classes: set only for graph classification
    :type num_classes: Optional[int](default = None)

    Example
    _______
    device = torch.device('cuda')
    model = GAT(task="node_classification", in_channels=18, hid_channels=32, out_channels=6, out_head=1).to(device)
    """

    def __init__(
        self,
        task: str,
        in_channels: int,
        hid_channels: int,
        out_channels: int,
        in_head: int = 4,
        out_head: int = 6,
        dropout: float = 0.0,
        num_classes: Optional[int] = None,
    ) -> None:
        super(GAT, self).__init__()

        assert task in ["node_classification", "graph_classification"]

        self.in_channels = in_channels
        self.hid_channels = hid_channels
        self.out_channels = out_channels
        self.dropout = dropout
        self.task = task

        self.gat1 = GATv2Conv(
            in_channels=in_channels,
            out_channels=hid_channels,
            heads=in_head,
            dropout=dropout,
        )
        self.relu1 = nn.ReLU(inplace=True)
        self.gat2 = GATv2Conv(
            in_channels=in_head * hid_channels,
            out_channels=hid_channels,
            heads=in_head,
            dropout=dropout,
        )
        self.relu2 = nn.ReLU(inplace=True)

        self.gat3 = GATv2Conv(
            in_channels=in_head * hid_channels,
            out_channels=hid_channels,
            heads=in_head,
            dropout=dropout,
        )
        self.relu3 = nn.ReLU(inplace=True)

        if task == "graph_classification":
            assert num_classes is not None
            self.gat4 = GATv2Conv(
                in_channels=in_head * hid_channels,
                out_channels=out_channels,
                heads=out_head,
                dropout=dropout,
                concat=False,
            )
            self.lin = nn.Linear(out_channels, num_classes)
        else:
            assert num_classes is None
            self.gat4 = GATv2Conv(
                in_channels=in_head * hid_channels,
                out_channels=out_channels,
                heads=1,
                dropout=dropout,
                concat=False,
            )

        self.apply(init_weights)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = self.gat1(x, edge_index)
        x = self.relu1(x)
        x = self.gat2(x, edge_index)
        x = self.relu2(x)
        x = self.gat3(x, edge_index)
        x = self.relu3(x)
        x = self.gat4(x, edge_index)

        if self.task == "graph_classification":
            x = global_mean_pool(x, batch)
            x = self.lin(x)

        return x
