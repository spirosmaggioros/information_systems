Train #1 for ENZYMES with GCN:
    - test_size = 0.25
    - num_layers 4
    - hidden_channels 256
    - out_channels 128
    - dropout 0.5
    - batch_size 2
    - epochs 3000
    - patience 100
    - model_name gcn_enzymes_1.pth

Results: Finished at epoch 296.
- Epoch: 196 | 
- train_acc: 0.5938
- train_auc: 0.8495
- train_recall: 0.5938
- train_precision: 0.5970
- train_specificity: 0.9187
- train_f1: 0.5946
- test_acc: 0.5583
- test_auc: 0.7832
- test_recall: 0.5583 
- test_precision: 0.5724
- test_specificity: 0.9117
- test_f1: 0.5471
- test_confussion_matrix: [[ 6  4  3  4  1  2]
                           [ 0  9  2  4  0  5]
                           [ 0  1 14  1  4  0]
                           [ 2  1  0 16  1  0]
                           [ 0  1  1  4  9  5]
                           [ 2  1  0  4  0 13]]


Just to note that if i use less dropout(i.e. 0.3 instead of 0.5) we have more overfitting, but better training results. We decided to provide the best non-overfitting model instead.

