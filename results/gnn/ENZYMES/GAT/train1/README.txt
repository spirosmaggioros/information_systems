Train #1 for ENZYMES with GAT:
    - test_size 0.25
    - num_layers 2
    - hidden_channels 64
    - out_channels 128
    - dropout 0.5
    - batch_size 2
    - epochs 1000
    - patience 100
    - model_name gat_enzymes_1.pth

Training finished at epoch 398.
- train_acc: 0.8604
- train_auc: 0.9825
- train_recall: 0.8604
- train_precision: 0.8627
- train_specificity: 0.9721
- train_f1: 0.8608
- test_acc: 0.6667
- test_auc: 0.8435
- test_recall: 0.6667
- test_precision: 0.7446
- test_specificity: 0.9333
- test_f1: 0.6708
- test_confussion_matrix: [[11  1  2  6  0  0]
                           [ 0  8  1  9  0  2]
                           [ 1  0 13  4  1  1]
                           [ 0  0  0 20  0  0]
                           [ 1  1  0  2 15  1]
                           [ 0  1  0  5  1 13]]


Note: We can clearly see the overfitting, i tried to reduce it as much as possible with setting dropout
to 0.5 and reducing the hidden layer size a lot, but still, overfitting is there, even with 2 layers in total.
