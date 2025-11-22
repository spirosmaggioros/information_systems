# Analysis on modern Graph Representation Learning and Graph Neural Network models

In this repo, you can train a GRL or GNN model on OGB datasets, perform inference on your own graphs or analysis on your computed graph embeddings. Our full documentation is hosted at: [TODO]

## Included models
  - [X] Graph2Vec
  - [X] NetLSD
  - [X] DeepWalk
  - [X] GCN
  - [X] GAT
  - [X] GIN
  - [X] PNA

For embedding analysis we also include code for t-SNE, Isomap, UMAP and k-d tree visualizations.

## Installation
To install as a pypi package, in the root directory, just do:
```bash
pip3 install .
```
Package name is: **information_systems**, so make sure to check if this exists.

## Example:

### Train a GRL model
```python
from dataloader.dataloader import ds_to_graphs
from ml_models.graph_models.graph2vec import Graph2Vec
from trainer.graph_trainer import train as train_graph

from visualizations.manifold import visualize_embeddings_manifold

data = ds_to_graphs("data/MUTAG")
best_model, metrics = train_graph(
    graph_model="graph2vec",
    graphs=data["graphs"],
    labels=data["graph_classes"],
    num_classes=6,
    mode="multiclass",
    out_channels=256,
    epochs=100,
    test_size=0.2,
    classifier="SVM",
    model_name="graph2vec_best.pkl",
    device="cpu",
)
embeddings = best_model.predict(data["graphs"])

visualize_embeddings_manifold(
    features=embeddings,
    labels=data["graph_classes"],
    method="TSNE",
    n_components=2,
    save_to="tsne_results.png"
)
```

### Train a GNN model
```python
import torch
from torch import nn

from dataloader.dataloader import ds_to_graphs, create_graph_dataloaders
from ml_models.torch_geometric.gat import GAT
from trainer.dl_trainer import train as train_torch_geometric

from visualizations.manifold import visualize_embeddings_manifold

data = ds_to_graphs("data/ENZYMES")

train_dataloader, test_dataloader = create_graph_dataloaders(
    data=data["graphs"],
    labels=data["graph_classes"],
    batch_size=2,
    test_size=0.2,
    device="cpu",
)

model = GAT(
    task="graph_classification",
    in_channels=18,
    hid_channels=128,
    out_channels=128,
    num_classes=6
)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.1, amsgrad=True)

training_results, best_checkpoint = train_torch_geometric(
    model=model,
    model_type="torch_geometric",
    train_dataloader=train_dataloader,
    test_dataloader=test_dataloader,
    loss_fn=loss_fn,
    optimizer=optimizer,
    mode="multiclass",
    num_classes=6,
    epochs=1000,
    patience=100,
    device="cpu",
    model_name="gat_best.pth",
    save_model=True,
)

from ml_models.torch_geometric.inference import inference as torch_geometric_inference

y_features, y_preds = torch_geometric_inference(
    model=model,
    mode="multiclass",
    dataloader=test_dataloader,
    mdoel_weights="gat_best.pth",
    out_json="inference_res.json",
)

# you can perform analysis now with the same code as in the GRL example
```

## CLI

We also implemented a CLI so you don't have to write code every time you need to train, perform inference or even perform simple analysis. You can do **information_systems --help** for more information, but we will list some examples below:

### Train a model:
```bash

information_systems train --model graph2vec \
                          --dataset_dir data/MUTAG \
                          --test_size 0.2 \
                          --classifier SVM \
                          --device cpu \
                          --out_channels 256 \
                          --epochs 100 \
                          --model_name graph2vec_model.pkl \
```

```bash
information_systems train --model gat \
                          --dataset_dir data/ENZYMES \
                          --test_size 0.2 \
                          --classifier SVM \
                          --hidden_channels 256 \
                          --out_channels 256 \
                          --dropout 0.3 \
                          --batch_size 2 \
                          --epochs 1000 \
                          --patience 100 \
                          --model_name gat_best.pth \
```

### Perform inference with a pre-trained model
```bash
information_systems inference --model graph2vec \
                              --dataset_dir data/MUTAG \
                              --model_weights graph2vec_model.pkl \
                              --out_json graph2vec_inference.json \
```

```bash
information_systems inference --model gat \
                              --dataset_dir data/ENZYMES \
                              --model_weights gat_best.pth \
                              --out_json gat_inference.json \
```

### Perform analysis on output embeddings
```bash
information_systems analysis --in_jsons gat_inference.json \
                             --manifold TSNE
```
