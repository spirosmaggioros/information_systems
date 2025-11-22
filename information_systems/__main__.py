import argparse
import json
from typing import Any

import torch
from torch import nn

from dataloader.dataloader import (
    add_random_edges,
    create_graph_dataloaders,
    ds_to_graphs,
    remove_random_edges,
    shuffle_node_attributes,
)
from ml_models.graph_models.deepwalk import DeepWalk
from ml_models.graph_models.graph2vec import Graph2Vec
from ml_models.graph_models.inference import inference as graph_model_inference
from ml_models.graph_models.netLSD import NetLSD
from ml_models.torch_geometric.gat import GAT
from ml_models.torch_geometric.gcn import GCN
from ml_models.torch_geometric.gin import _GIN
from ml_models.torch_geometric.inference import inference as torch_geometric_inference
from trainer.clustering_trainer import train as clustering_trainer
from trainer.dl_trainer import train as dl_trainer
from trainer.graph_trainer import train as graph_trainer
from visualization.clustering import scatter_clusters
from visualization.manifold import visualize_embeddings_manifold

GRAPH_MODELS = ["graph2vec", "netlsd", "deepwalk"]
TORCH_GEOMETRIC_MODELS = ["gcn", "gat", "gin"]
CLASSIFIERS = ["MLP", "SVM"]


def get_ml_model(
    model_name: str,
    in_channels: int,
    hidden_channels: int,
    out_channels: int,
    dropout: float,
    num_classes: int,
) -> nn.Module:
    name_to_model = {
        "gcn": GCN(
            task="graph_classification",
            in_channels=in_channels,
            hid_channels=hidden_channels,
            out_channels=out_channels,
            dropout=dropout,
            num_classes=num_classes,
        ),
        "gat": GAT(
            task="graph_classification",
            in_channels=in_channels,
            hid_channels=hidden_channels,
            out_channels=out_channels,
            dropout=dropout,
            num_classes=num_classes,
        ),
        "gin": _GIN(
            task="graph_classification",
            in_channels=in_channels,
            hid_channels=hidden_channels,
            out_channels=out_channels,
            dropout=dropout,
            num_classes=num_classes,
        ),
        "graph2vec": Graph2Vec(
            dimensions=out_channels,
        ),
        "netlsd": NetLSD(),
        "deepwalk": DeepWalk(dimensions=out_channels),
    }

    return name_to_model[model_name]


def run_train(args: Any) -> None:

    assert (
        args.model in GRAPH_MODELS or args.model in TORCH_GEOMETRIC_MODELS
    ), "Please select an available model"
    assert args.test_size < 1.0 and args.test_size > 0.0
    assert args.device in ["cpu", "mps", "cuda"]

    data = ds_to_graphs(dataset_folder=args.dataset_dir)
    graphs = data["graphs"]
    labels = data["graph_classes"]
    node_attributes = data["node_attributes"]

    num_classes = len(set(labels))

    model = get_ml_model(
        model_name=args.model,
        in_channels=(
            len(node_attributes[list(node_attributes.keys())[0]])
            if len(node_attributes) > 0
            else 0
        ),
        hidden_channels=args.hidden_channels,
        out_channels=args.out_channels,
        dropout=args.dropout,
        num_classes=1 if num_classes == 2 else num_classes,
    )

    if args.shuffle_node_attributes:
        for graph in graphs:
            shuffle_node_attributes(graph)
    if args.add_random_edges > 0.0:
        assert (
            args.remove_random_edges == 0.0
        ), "We suggest to notuse add_random_edges or remove_random_edges together for stability analysis"
        for graph in graphs:
            add_random_edges(graph, args.add_random_edges)
    if args.remove_random_edges > 0.0:
        for graph in graphs:
            remove_random_edges(graph, args.remove_random_edges)

    if args.model in GRAPH_MODELS:
        assert (
            args.classifier in CLASSIFIERS
        ), "Please select an available classifier model"
        assert args.epochs > 0

        _, metrics = graph_trainer(
            graph_model=args.model,
            graphs=graphs,
            labels=labels,
            num_classes=1 if num_classes == 2 else num_classes,
            mode="binary" if num_classes == 2 else "multiclass",
            out_channels=args.out_channels,
            epochs=args.epochs,
            test_size=args.test_size,
            classifier=args.classifier,
            model_name=args.model_name,
            device=args.device,
        )
    else:
        assert args.epochs > 0
        assert args.patience > 0
        train_dataloader, test_dataloader = create_graph_dataloaders(
            data=graphs,
            labels=labels,
            batch_size=args.batch_size,
        )

        loss_fn = None
        if num_classes == 2:
            loss_fn = nn.BCEWithLogitsLoss()
        else:
            loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=0.001, weight_decay=0.1, amsgrad=True
        )

        _, metrics = dl_trainer(
            model=model,
            model_type="torch_geometric",
            mode="binary" if num_classes == 2 else "multiclass",
            train_dataloader=train_dataloader,
            test_dataloader=test_dataloader,
            num_classes=1 if num_classes == 2 else num_classes,
            loss_fn=loss_fn,
            optimizer=optimizer,
            epochs=args.epochs,
            patience=args.patience,
            device=args.device,
            target_dir=args.target_dir,
            model_name=args.model_name,
            save_model=True,
        )

    print(f"Training metrics: {metrics}")


def run_inference(args: Any) -> None:
    data = ds_to_graphs(args.dataset_dir)
    graphs = data["graphs"]
    labels = data["graph_classes"]
    node_attributes = data["node_attributes"]
    num_classes = len(set(labels))

    model = get_ml_model(
        model_name=args.model,
        in_channels=(
            len(node_attributes[list(node_attributes.keys())[0]])
            if len(node_attributes) > 0
            else 0
        ),
        hidden_channels=args.hidden_channels,
        out_channels=args.out_channels,
        dropout=args.dropout,
        num_classes=1 if num_classes == 2 else num_classes,
    )

    dataloader = create_graph_dataloaders(
        graphs,
        labels,
        batch_size=1,
        test_size=0.0,
    )

    if args.model in ["graph2vec", "netlsd", "deepwalk"]:
        graph_model_inference(
            model=model,
            data=graphs,
            model_weights=args.model_weights,
            out_json=args.out_json,
            ground_truth_labels=labels,
        )
    else:
        torch_geometric_inference(
            model=model,
            model_weights=args.model_weights,
            dataloader=dataloader,
            out_json=args.out_json,
            ground_truth_labels=labels,
            mode="graph_classification",
        )

    print(f"Results saved at {args.out_json}")


def run_analysis(args: Any) -> None:
    files = args.in_jsons

    if len(files) > 1:
        pass
    else:
        file = files[0]

        with open(file, "r") as f:
            data = json.load(f)
            features = data["out_features"]
            if "predictions" in data.keys():
                predictions = data["predictions"]
            else:
                predictions = []
            if "y_hat" in data.keys():
                labels = data["y_hat"]
            else:
                labels = []

        if len(args.clustering) != 0:
            best_model, stats = clustering_trainer(
                model_type=args.clustering,
                graph_embeddings=features,
                labels=labels if len(labels) > 0 else predictions,
                num_classes=len(set(labels)),
            )

            scatter_clusters(
                model=best_model,
                data=features,
            )

            print(f"ARI score: {stats['ARI']}")

        if len(args.manifold) != 0:
            visualize_embeddings_manifold(
                method=args.manifold,
                features=features,
                labels=labels if len(labels) > 0 else predictions,
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="information_systems",
        description="In this repo, you can train a GRL or GNN model on OGB datasets, perform inference on your own graphs or analysis on your computed graph embeddings.",
        usage="""
        Train a model:
        information_systems train --model graph2vec \
                                  --dataset_dir data/MUTAG \
                                  --test_size 0.2 \
                                  --classifier SVM \
                                  --device cpu \
                                  --out_channels 256 \
                                  --epochs 100 \
                                  --model_name graph2vec_model.pkl \

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

        Perform inference with a pre-trained model:
        information_systems inference --model graph2vec \
                                      --dataset_dir data/MUTAG \
                                      --model_weights graph2vec_model.pkl \
                                      --out_json graph2vec_inference.json \

        information_systems inference --model gat \
                                      --dataset_dir data/ENZYMES \
                                      --model_weights gat_best.pth \
                                      --out_json gat_inference.json \

        Perform analysis with output embeddings:
        information_systems analysis --in_jsons gat_inference.json \
                                     --manifold TSNE""",
        add_help=True,
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # train a graph representation learning model or a torch geometric model
    train = subparsers.add_parser("train", help="Selects training mode")
    train.add_argument(
        "--model",
        type=str,
        required=True,
        help="[REQUIRED] Select model you want to train. Please read README to see currently supported models",
    )

    train.add_argument(
        "--dataset_dir",
        type=str,
        required=True,
        help="[REQUIRED] Select a dataset to train on. We currently support a specific amount of datasets, please read README for more information",
    )

    train.add_argument(
        "--test_size",
        type=float,
        default=0.2,
        required=False,
        help="Specify the percentage of data that will be used for testing(default = 0.2)",
    )

    train.add_argument(
        "--classifier",
        type=str,
        default="SVM",
        required=False,
        help="Specify the classifier(either svm or mlp)",
    )

    train.add_argument(
        "--device",
        type=str,
        default="cpu",
        required=False,
        help="Specify the device to train(only for torch geometric, either cpu, mps or cuda). Note that mps might have some bugs",
    )

    train.add_argument(
        "--hidden_channels",
        type=int,
        default=256,
        required=False,
        help="Only for torch geometric: Specify the number of hidden channels",
    )

    train.add_argument(
        "--out_channels",
        type=int,
        default=256,
        required=False,
        help="Only for torch geometric: Specify the dimensionality of output features",
    )

    train.add_argument(
        "--dropout",
        type=float,
        default=0.0,
        required=False,
        help="Only for torch geometric: Specify the dropout value for each convolutional layer of torch geometric models",
    )

    train.add_argument(
        "--batch_size",
        type=int,
        required=False,
        default=2,
        help="Only for torch geometric: Specify the batch size for training",
    )

    train.add_argument(
        "--epochs",
        type=int,
        default=200,
        required=False,
        help="Only for torch geometric: Specify the number of epochs the model will train",
    )

    train.add_argument(
        "--patience",
        type=int,
        default=10,
        required=False,
        help="Only for torch geometric: Specify the number of epochs the model will train without improvement",
    )

    train.add_argument(
        "--target_dir",
        type=str,
        default=".",
        required=False,
        help="Only for torch geometric: Specify the directory that the model weights will be saved in",
    )

    train.add_argument(
        "--model_name",
        type=str,
        default="dl_trainer_best_model.pth",
        required=False,
        help="Only for torch geometric: Specify the name of saved model weights",
    )

    train.add_argument(
        "--shuffle_node_attributes",
        action="store_true",
        help="If set, then node attributes for each graph in the dataset will be shuffled. Used for embedding stability analysis",
    )

    train.add_argument(
        "--add_random_edges",
        type=float,
        default=0.0,
        help="If set, a random p%(only new edges) of the total edges of each graph will be added",
    )

    train.add_argument(
        "--remove_random_edges",
        type=float,
        default=0.0,
        help="If set, a random p% of the total edges of each graph will be removed",
    )
    train.set_defaults(func=run_train)

    # Perform inference with a pre-trained model(Used to extract embeddings for all other tasks)
    inference = subparsers.add_parser("inference", help="Selects inference mode")

    inference.add_argument(
        "--dataset_dir",
        type=str,
        required=True,
        help="[REQUIRED] Select a dataset to train on. We currently support a specific amount of datasets, please read README for more information",
    )

    inference.add_argument(
        "--model",
        type=str,
        required=True,
        help="[REQUIRED] Input model name for inference",
    )

    inference.add_argument(
        "--model_weights",
        type=str,
        required=True,
        help="[REQUIRED] Absolute path of the model weights",
    )

    inference.add_argument(
        "--hidden_channels",
        type=int,
        default=256,
        required=False,
        help="Only for torch geometric: Specify the number of hidden channels",
    )

    inference.add_argument(
        "--out_channels",
        type=int,
        default=256,
        required=False,
        help="Only for torch geometric: Specify the dimensionality of output features",
    )

    inference.add_argument(
        "--dropout",
        type=float,
        default=0.0,
        required=False,
        help="Only for torch geometric: Specify the dropout value for each convolutional layer of torch geometric models",
    )

    inference.add_argument(
        "--out_json",
        type=str,
        required=True,
        help="[REQUIRED] Absolute path of the .json file with predictions and out features",
    )
    inference.set_defaults(func=run_inference)

    # perform analysis on predictions and features from inference
    analysis = subparsers.add_parser("analysis", help="Selects analysis mode")

    analysis.add_argument(
        "--in_jsons",
        nargs="+",
        required=True,
        help="[REQUIRED] Specify json files(can be > 1) to perform analysis on. These files should have predictions and out_features key values. \
              Multiple json files are only needed for stability analysis tasks.",
    )

    analysis.add_argument(
        "--clustering",
        type=str,
        default="",
        required=False,
        help="Select an available clustering method to perform clustering on passed json file. Note that to perform this, you should only pass one json file",
    )

    analysis.add_argument(
        "--manifold",
        type=str,
        default="",
        required=False,
        help="Select an available manifold method to perform analysis on",
    )

    analysis.add_argument(
        "--stability_analysis",
        action="store_true",
        required=False,
        help="Set to true to perform stability analysis. You have to provide 2 json files for this to happen.",
    )
    analysis.set_defaults(func=run_analysis)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
