import torch
import torch.nn as nn
import torch.nn.functional as F

class LinearQNet(nn.Module):
    '''
    A deep linear Q-net which can generate nets of arbitrary shape.

    :param input_size: (int) Amount of input parameters the net accepts
    :param hidden_shape: (list[int]) The ith element describes the amount of neurons the ith hidden layer contains.
    :param output_size: (int) Amount of output parameters returned.
    '''
    def __init__(self, input_size: int, hidden_shape: list, output_size: int) -> None:
        super().__init__() 

        # Making a linear net of arbitrarily many hidden layers.
        self.net = nn.ModuleList([nn.Linear(input_size, hidden_shape[0])])
        for i in range(len(hidden_shape) - 1):
            self.net.append(nn.Linear(hidden_shape[i], hidden_shape[i + 1]))
        self.net.append(nn.Linear(hidden_shape[-1], output_size))

    def forward(self, x: torch.tensor) -> torch.tensor:
        '''
        Feeding an input through the net.

        :param x: (torch.tensor) Input tensor of dimension input_size.
        :return: (torch.tensor) A tensor output of dimension output_size from the net. 
        '''
        # Iteratively feeding it through each layer.
        for i, layer in enumerate(self.net):
            x = layer(x)

            # We don't feed it through a ReLU if it's at the output layer.
            if i < len(self.net) - 1:
                x = F.relu(x)

        return x
        