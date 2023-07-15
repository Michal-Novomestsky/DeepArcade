import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

from pathlib import Path

class LinearQNet(nn.Module):
    '''
    A deep linear Q net which can generate nets of arbitrary shape.

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
    
class QTrainer:
    def __init__(self, net: nn.Module, learning_rate: float, discount_rate: float) -> None:
        self.discount_rate = discount_rate
        self.net = net

        # Using the Adam optimiser with a mean-squared error loss function.
        self.optimiser = optim.Adam(net.parameters(), lr=learning_rate)
        self.loss_function = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, gameover):
        prediction = self.net(state)
        target = prediction.clone()

        for i in range(len(gameover)):
            if gameover[i]:
                # If there is no next state (i.e. the game ended), don't update the Q-value (since there isn't any new data to compare with).
                Q_new = reward[i]
            else:
                # Q-value which maximises the expected reward
                Q_new = reward[i] + self.discount_rate*torch.max(self.net(next_state[i])).cuda()

            # Updating the output node (i.e. the one that produced the Q-value used to calculate Q_new) with Q_new.
            target[i][torch.argmax(action).item()] = Q_new
        
        # Calculating loss and running backprop.
        self.optimiser.zero_grad()
        loss = self.loss_function(target, prediction)
        loss.backward()
        self.optimiser.step()
        