#!/bin/bash

information_systems train --model graph2vec \
                          --dataset_dir data/IMDB-MULTI \
                          --test_size 0.2 \
                          --classifier SVM \
                          --device cpu \
                          # --hidden_channels 256 \
                          --out_channels 256 \
                          # --dropout 0.3 \
                          # --batch_size 2 \
                          --epochs 100 \
                          # --patience 100 \


