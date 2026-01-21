#!/bin/bash

python3.10 -m information_systems train_classifier \
                          --classifier SVC \
                          --dataset_dir data/MUTAG \
                          --out_embeddings graph2vec_mutag_inference.json \
                          --test_size 0.25 \
                          --device mps \
                          --model_name svm_graph2vec_mutag.pkl \
                          --n_trials 1500 \
