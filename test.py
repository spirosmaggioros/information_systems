import torch
from torch import nn

from dataloader.dataloader import create_graph_dataloaders, ds_to_graphs

# from torch_geometric.nn import GAT, global_mean_pool
# from ml_models.torch_geometric.gat import GAT
from ml_models.torch_geometric.gin import _GIN
from trainer.dl_trainer import train as dl_train

# from visualization.graph_visualizations import plot_graph


if __name__ == "__main__":
    data = ds_to_graphs("data/ENZYMES")

    # model = GAT(
    #     task="graph_classification",
    #     in_channels=18,
    #     hid_channels=128,
    #     out_channels=128,
    #     num_classes=6,
    #     dropout=0.3,
    # )

    model = _GIN(
        task="graph_classification",
        in_channels=18,
        hid_channels=128,
        out_channels=128,
        num_classes=6,
        dropout=0.3,
    )

    # class _GAT(nn.Module):
    #     def __init__(self) -> None:
    #         super().__init__()
    #         self.model = GAT(
    #             in_channels=18,
    #             hidden_channels=128,
    #             out_channels=128,
    #             num_layers=4,
    #             v2=True,
    #         )
    #         self.lin = nn.Linear(128, 6)

    #     def forward(self, x, edge_index, batch) -> torch.Tensor:
    #         x = self.model(x, edge_index, batch)
    #         x = global_mean_pool(x, batch)
    #         return self.lin(x)

    train_dataloader, test_dataloader = create_graph_dataloaders(
        data=data["graphs"],
        labels=data["graph_classes"],
        batch_size=2,
        device="cpu",
    )

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=0.001, weight_decay=0.1, amsgrad=True
    )
    loss = nn.CrossEntropyLoss()

    results, best = dl_train(
        model=model,
        model_type="torch_geometric",
        mode="multiclass",
        num_classes=6,
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        loss_fn=loss,
        optimizer=optimizer,
        epochs=5000,
        patience=100,
        device="cpu",
        save_model=True,
    )

    print(results["training_time"], best)
