Train #2 for ENZYMES with GCN:
    - test_size = 0.25
    - num_layers 4
    - hidden_channels 256
    - out_channels 128
    - dropout 0.5
    - batch_size 2
    - epochs 3000
    - patience 100
    - model_name gcn_enzymes_2.pth

Training finished at epoch 255:
- train_acc: 0.5521
- train_auc: 0.8266
- train_recall: 0.5521
- train_precision: 0.5516
- train_specificity: 0.9104
- train_f1: 0.5505
- test_acc: 0.5417
- test_auc: 0.7778
- test_recall: 0.5417
- test_precision: 0.5653
- test_specificity: 0.9083
- test_f1: 0.5410
- test_confussion_matrix: [[ 9  2  1  2  2  4]
                           [ 0  9  0  5  0  6]
                           [ 0  1 12  5  2  0]
                           [ 1  1  1 13  3  1]
                           [ 3  2  1  3  8  3]
                           [ 3  1  0  2  0 14]]

