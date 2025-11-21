#!/bin/bash

information_systems train --model gat \
                          --dataset_dir data/ENZYMES \
                          --test_size 0.2 \
                          --classifier SVM \
                          --device cpu \
                          --hidden_channels 256 \
                          --out_channels 256 \
                          --dropout 0.3 \
                          --batch_size 2 \
                          --epochs 2000 \
                          --patience 100 \
                          --shuffle_node_attributes \


