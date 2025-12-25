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

Training finished at epoch 448

- train_acc: 0.7396
- train_auc: 0.9368
- train_recall: 0.7396
- train_precision: 0.7394
- train_specificity: 0.9479
- train_f1: 0.7389
- test_acc: 0.5917
- test_auc: 0.8292
- test_recall: 0.5917
- test_precision: 0.6507
- test_specificity: 0.9183
- test_f1: 0.5965
- test_confussion_matrix: [[11  3  0  2  1  3]
                           [ 0 12  0  0  0  8]
                           [ 5  0 11  0  0  4]
                           [ 1  3  0 15  0  1]
                           [ 6  2  1  1  8  2]
                           [ 0  3  0  2  1 14]]
