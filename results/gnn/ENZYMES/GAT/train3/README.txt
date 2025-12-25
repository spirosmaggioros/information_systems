Train 31 for ENZYMES with GAT:
    - test_size 0.25
    - num_layers 2
    - hidden_channels 64
    - out_channels 128
    - dropout 0.5
    - batch_size 2
    - epochs 1000
    - patience 100
    - model_name gat_enzymes_3.pth


Training finished at epoch 429


- train_acc: 0.8750
- train_auc: 0.9822
- train_recall: 0.8750
- train_precision: 0.8748
- train_specificity: 0.9750
- train_f1: 0.8748
- test_acc: 0.6833
- test_auc: 0.8523
- test_recall: 0.6833
- test_precision: 0.7123
- test_specificity: 0.9367
- test_f1: 0.6895
- test_confussion_matrix: [[12  0  1  1  4  2]
                           [ 1 12  0  2  1  4]
                           [ 0  0 16  2  0  2]
                           [ 4  0  0 15  1  0]
                           [ 4  0  0  1 14  1]
                           [ 3  1  0  2  1 13]]

