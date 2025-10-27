import torch
from torch import nn
from torch_geometric.Data import Data
from torch_geometric.nn import GCNConv, Sequential


class GCN(nn.Module):

    def __init__(
        self,
        in_channels: int,
        hid_channels: int,
        out_channels: int,
        dropout: float = 0.5,
    ) -> None:
        super(GCN, self).__init__()

        self.in_channels = in_channels
        self.hid_channels = hid_channels
        self.out_channels = out_channels
        self.dropout = dropout

        self.gcn_net = Sequential(
            "x, edge_index",
            [
                (
                    GCNConv(
                        in_channels=in_channels,
                        out_channels=hid_channels,
                        dropout=dropout,
                    ),
                    "x, edge_index -> x",
                ),
                nn.ReLU(inplace=True),
                (
                    GCNConv(
                        in_channels=hid_channels,
                        out_channels=out_channels,
                        dropout=dropout,
                    ),
                    "x, edge_index -> x",
                ),
                nn.ReLU(inplace=True),
            ],
        )

    def forward(self, data: Data) -> torch.Tensor:
        x, edge_index = data.x, data.edge_index
        return self.gcn_net(x, edge_index)
