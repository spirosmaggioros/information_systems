from dataloader.dataloader import ds_to_graphs
from trainer.graph2vec_trainer import train as graph2vec_train

# from visualization.graph_visualizations import plot_graph

data = ds_to_graphs("data/IMDB-MULTI")

param_grid = {
    "dimensions": [64, 128, 256],
    "wl_iterations": [2, 3],
    "epochs": [100],
}

model, metrics = graph2vec_train(
    data["graphs"], data["graph_classes"], classifier="MLP", num_classes=3, device="cpu"
)

print(metrics)


# print(f"Best parameters: {best_params}")
