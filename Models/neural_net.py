import torch
import torch.nn as nn
import torch.nn.functional as F

from pathlib import Path

class LinearQNet(nn.Module):
    '''
    A deep linear Q-net which can generate nets of arbitrary shape.

    :param input_size: (int) Amount of input parameters the net accepts
    :param hidden_size: (list[int]) The ith element describes the amount of neurons the ith hidden layer contains.
    :param output_size: (int) Amount of output parameters returned.
    '''
    def __init__(self, input_size: int, hidden_size: list, output_size: int) -> None:
        super().__init__()
        
        # Making a linear net of arbitrarily many hidden layers.
        self.net = [nn.Linear(input_size, hidden_size[0])]
        for i in range(1, len(hidden_size)):
            self.net.append(nn.Linear(hidden_size[i - 1], hidden_size[i]))
        self.net.append(nn.Linear(hidden_size[-1], output_size))

    def forward(self, x) -> float:
        # Iteratively feeding it through each layer.
        for i, layer in enumerate(self.net):
            x = layer(x)

            # We don't feed it through a ReLU if it's at the output layer.
            if i < len(self.net) - 1:
                x = F.relu(x)

        return x
        