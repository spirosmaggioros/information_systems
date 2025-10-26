import torch.nn as nn


def init_weights(net: nn.Module) -> None:
    for block in net:
        for layer in block.children():
            for name, weight in layer.named_parameters():
                if "weight" in name:
                    nn.init.kaiming_normal_(weight)
                if "bias" in name:
                    nn.init.constant_(weight, 0.0)
