#!/bin/bash

information_systems inference --model graph2vec \
                              --dataset_dir data/ENZYMES \
                              --model_weights graph2vec_enzymes_svm_1.pkl \
                              --out_json graph2vec_inference_enzymes.json \
                              --num_layers 2 \
                              --hidden_channels 64 \
                              --out_channels 128 \
                              --dropout 0.5 \


