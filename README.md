# Analysis on modern Graph Representation Learning and Graph Neural Network models

In this repo, you can train a GRL or GNN model on graph datasets (TU Dataset format), perform inference on your own graphs, conduct robustness analysis, or analyze your computed graph embeddings.

## Included models
  - [X] Graph2Vec
  - [X] NetLSD
  - [X] DeepWalk
  - [X] GCN
  - [X] GAT
  - [X] GIN
  - [X] PNA


For embedding analysis we also include code for t-SNE and Isomap visualizations.

> **Note:** UMAP support is available but requires manual installation due to dependency conflicts with `karateclub`. See the visualization module for details.

## Classification Models
For downstream classification on graph embeddings:
- [X] SVM (Support Vector Machine with RBF/Linear kernels)
- [X] MLP (Multi-Layer Perceptron)

## Clustering Models
For unsupervised analysis of graph embeddings:
- [X] K-Means
- [X] Spectral Clustering

## Visualization Tools
- [X] t-SNE manifold visualization
- [X] Isomap manifold visualization
- [X] Cluster scatter plots
- [X] Graph structure visualization (with node/edge labels)

## Perturbation & Robustness Analysis
Tools for testing model robustness under graph perturbations:
- [X] Random edge addition
- [X] Random edge removal
- [X] Node attribute shuffling

## Evaluation Metrics
All training runs report:
- Accuracy
- AUROC (Area Under ROC Curve)
- F1 Score (macro-averaged for multiclass)
- Precision
- Recall
- Specificity
- Confusion Matrix
- Peak Memory Usage
- Training Time

## Supported Dataset Format
This package supports datasets in the **TU Dataset format** (e.g., MUTAG, ENZYMES, PROTEINS, IMDB-MULTI). The dataset folder should contain:
- `*_A.txt` - Edge list
- `*_graph_indicator.txt` - Graph membership for each node
- `*_graph_labels.txt` - Graph class labels
- `*_node_labels.txt` (optional) - Node labels
- `*_node_attributes.txt` (optional) - Node feature vectors
- `*_edge_labels.txt` (optional) - Edge labels

Example structure:
data/MUTAG/
├── MUTAG_A.txt
├── MUTAG_graph_indicator.txt
├── MUTAG_graph_labels.txt
├── MUTAG_node_labels.txt
└── MUTAG_edge_labels.txt

## Installation
To install as a pypi package, in the root directory, just do:
```bash
pip3 install .
```
Package name is: **information_systems**, so make sure to check if this exists.

## CLI

We implemented a CLI so you don't have to write code every time you need to train, perform inference or even perform simple analysis. You can do **information_systems --help** for more information, but we will list some examples below:

### Train a model:
```bash

information_systems train --model graph2vec \
                          --dataset_dir data/MUTAG \
                          --test_size 0.25 \
                          --classifier SVM \
                          --out_channels 256 \
                          --epochs 100 \
                          --model_name graph2vec_model.pkl \
                          --device cpu \
```

```bash
information_systems train --model gat \
                          --dataset_dir data/ENZYMES \
                          --test_size 0.25 \
                          --classifier SVM \
                          --hidden_channels 64 \
                          --out_channels 128 \
                          --dropout 0.5 \
                          --batch_size 2 \
                          --epochs 1000 \
                          --patience 100 \
                          --model_name gat_best.pth \
```

### Perform inference with a pre-trained model
```bash
information_systems inference --model graph2vec \
                              --out_channels 128 \
                              --dataset_dir data/MUTAG \
                              --model_weights graph2vec_model.pkl \
                              --out_json graph2vec_inference.json \
```

```bash
information_systems inference --model gat \
                              --num_layers 2 \
                              --hidden_channels 64 \
                              --out_channels 128 \
                              --dropout 0.5 \
                              --dataset_dir data/ENZYMES \
                              --model_weights gat_best.pth \
                              --out_json gat_inference.json \
```

### Perform inference with perturbations
```bash
# Add 20% random edges
information_systems inference --model gin \
                              --dataset_dir data/ENZYMES \
                              --model_weights gin_best.pth \
                              --out_json gin_perturbed.json \
                              --add_random_edges 0.2

# Remove 15% random edges
information_systems inference --model graph2vec \
                              --out_channels 128 \
                              --dataset_dir data/MUTAG \
                              --model_weights graph2vec_model.pkl \
                              --out_json graph2vec_perturbed.json \
                              --remove_random_edges 0.15

# Shuffle node attributes
information_systems inference --model gat \
                              --num_layers 2 \
                              --hidden_channels 64 \
                              --out_channels 128 \
                              --dataset_dir data/ENZYMES \
                              --model_weights gat_best.pth \
                              --out_json gat_shuffled.json \
                              --shuffle_node_attributes
```

### Perform analysis on output embeddings
```bash
information_systems analysis --in_jsons gat_inference.json \
                             --manifold TSNE
```

### Perform clustering analysis on embeddings
```bash
information_systems analysis --in_jsons gat_inference.json \
                             --clustering kmeans
                             
information_systems analysis --in_jsons gat_inference.json \
                             --clustering spectral
```
