Train #3 for ENZYMES with GCN:
    - test_size = 0.25
    - num_layers 4
    - hidden_channels 256
    - out_channels 128
    - dropout 0.5
    - batch_size 2
    - epochs 3000
    - patience 100
    - model_name gcn_enzymes_3.pth

Training finished at epoch 280.

- train_acc: 0.5896
- train_auc: 0.8437
- train_recall: 0.5896
- train_precision: 0.5894
- train_specificity: 0.9179
- train_f1: 0.5887
- test_acc: 0.5750
- test_auc: 0.7849
- test_recall: 0.5750
- test_precision: 0.6252
- test_specificity: 0.9150
- test_f1: 0.5799
- test_confussion_matrix: [[10  4  0  2  1  3]
                           [ 0 10  0  4  0  6]
                           [ 0  1 11  2  4  2]
                           [ 1  2  0 10  7  0]
                           [ 0  1  1  0 14  4]
                           [ 1  2  0  2  1 14]]
