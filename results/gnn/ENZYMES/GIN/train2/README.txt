Tain #1 for ENZYMES with PNA:
    - test_size 0.25
    - num_layers 2
    - hidden_channels 64
    - out_channels 128
    - dropout 0.5
    - batch_size 2
    - epochs 1000
    - patience 100
    - model_name gin_enzymes_1.pth

Training finished at epoch 571

- train_acc: 0.7938
- train_auc: 0.9615
- train_recall: 0.7938
- train_precision: 0.7940
- train_specificity: 0.9588
- train_f1: 0.7932
- test_acc: 0.6417
- test_auc: 0.8610
- test_recall: 0.6417
- test_precision: 0.6460
- test_specificity: 0.9283
- test_f1: 0.6417
- test_confussion_matrix: [[ 9  3  2  1  3  2]
                           [ 1 14  0  1  0  4]
                           [ 2  0 14  0  4  0]
                           [ 1  2  0 15  1  1]
                           [ 3  1  1  0 14  1]
                           [ 2  1  2  1  3 11]]
