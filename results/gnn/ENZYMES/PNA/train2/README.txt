Tain #1 for ENZYMES with PNA:
    - test_size 0.25
    - num_layers 4
    - hidden_channels 64
    - out_channels 128
    - dropout 0.5
    - batch_size 2
    - epochs 1000
    - patience 100
    - model_name pna_enzymes_2.pth

Training finished at epoch 788

- train_acc: 0.7417
- train_auc: 0.9358
- train_recall: 0.7417
- train_precision: 0.7417
- train_specificity: 0.9483
- train_f1: 0.7416
- test_acc: 0.6667
- test_auc: 0.8652
- test_recall: 0.6667
- test_precision: 0.6718
- test_specificity: 0.9333
- test_f1: 0.6647
- test_confussion_matrix: [[13  1  2  2  1  1]
                           [ 1 17  0  0  0  2]
                           [ 2  3 11  2  0  2]
                           [ 1  1  3 13  1  1]
                           [ 1  3  2  2 12  0]
                           [ 0  2  1  1  2 14]]
