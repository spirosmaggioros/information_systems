#!/bin/bash

information_systems inference --model graph2vec \
                              --dataset_dir data/MUTAG \
                              --model_weights graph2vec_mutag_svm_1.pkl  \
                              --out_json graph2vec_mutag_inference.json \
                              --out_channels 128 \


