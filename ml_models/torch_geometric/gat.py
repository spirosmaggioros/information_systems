import torch
from torch import nn
from torch_geometric.data import Data
from torch_geometric.nn import GATConv, Sequential


class GAT(nn.Module):
    """
    Implementation of a Graph Attention Network

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
    :param dropout: the dropout value for GATConv
    :type dropout: float

    Example
    _______
    device = torch.device('cuda')
    model = GAT(in_channels=1433, hid_channels=8, out_channels=7).to(device)
    """

    def __init__(
        self,
        in_channels: int,
        hid_channels: int,
        out_channels: int,
        in_head: int = 8,
        out_head: int = 1,
        dropout: float = 0.5,
    ) -> None:
        super(GAT, self).__init__()

        self.in_channels = in_channels
        self.hid_channels = hid_channels
        self.out_channels = out_channels
        self.dropout = dropout

        self.gat_net = Sequential(
            "x, edge_index",
            [
                (
                    GATConv(
                        in_channels=in_channels,
                        out_channels=hid_channels,
                        heads=in_head,
                        dropout=dropout,
                    ),
                    "x, edge_index -> x",
                ),
                nn.ReLU(inplace=True),
                (
                    GATConv(
                        in_channels=hid_channels * in_head,
                        out_channels=out_channels,
                        heads=out_head,
                        dropout=dropout,
                        concat=False,
                    ),
                    "x, edge_index -> x",
                ),
                nn.ReLU(inplace=True),
            ],
        )

    def forward(self, data: Data) -> torch.Tensor:
        x, edge_index = data.x, data.edge_index
        return self.gat_net(x, edge_index)
