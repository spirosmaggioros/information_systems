Train #2 for ENZYMES with GAT:
    - test_size = 0.25
    - hidden_channels 64
    - out_channels 128
    - dropout 0.5
    - batch_size 2
    - epochs 3000
    - patience 100
    - model_name gat_enzymes_2.pth

Training finished at 282 epochs.

- train_acc: 0.8729
- train_auc: 0.9833
- train_recall: 0.8729
- train_precision: 0.8729
- train_specificity: 0.9746
- train_f1: 0.8727
- test_acc: 0.6500
- test_auc: 0.8522
- test_recall: 0.6500
- test_precision: 0.6847
- test_specificity: 0.9300
- test_f1: 0.6556
- test_confussion_matrix: [[12  3  0  2  2  1]
                           [ 0 13  0  4  1  2]
                           [ 0  0 14  2  1  3]
                           [ 1  1  1 16  0  1]
                           [ 2  2  0  2 12  2]
                           [ 0  2  0  6  1 11]]


