#!/bin/bash

information_systems train --model graph2vec \
                          --dataset_dir data/MUTAG \
                          --test_size 0.2 \
                          --classifier SVM \
                          --device cpu \
                          --out_channels 256 \
                          --epochs 100 \
                          --model_name graph2vec_model.pkl \


