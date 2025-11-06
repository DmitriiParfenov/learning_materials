from typing import Tuple

import torch.nn as nn


class RegressionModel(nn.Module):
    def __init__(self, n_features: int, n_hidden: Tuple[int, int]) -> None:
        super().__init__()
        self.input = nn.Linear(n_features, n_hidden[0])
        self.activation = nn.ReLU()
        self.layer_1 = nn.Linear(n_hidden[0], n_hidden[1])
        self.output = nn.Linear(n_hidden[1], 1)
        layers = [self.input, self.activation, self.layer_1, self.activation, self.output]
        self.module_list = nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.module_list:
            x = layer(x)
        return x
