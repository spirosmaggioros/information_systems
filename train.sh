information_systems train --model graph2vec \
                          --dataset_dir data/MUTAG \
                          --test_size 0.25 \
                          --classifier SVC \
                          --device mps \
                          --out_channels 128 \
                          --epochs 100 \
                          --model_name graph2vec_mutag_svm.pkl \
                          --classifier_name svm_mutag_graph2vec.pkl
