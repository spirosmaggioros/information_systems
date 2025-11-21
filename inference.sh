#!/bin/bash

information_systems inference --model gat \
                              --dataset_dir data/ENZYMES \
                              --model_weights dl_trainer_best_model.pth \
                              --out_json test_json.json \
                              --hidden_channels 256 \
                              --out_channels 256 \


