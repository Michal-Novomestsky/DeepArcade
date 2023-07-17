import torch
import torch.nn as nn
import torch.optim as optim

from Models.neural_net import *
from Games.snake_game import SnakeGameAI

class Environment():
    def __init__(self, input_size: int, hidden_size: list, output_size: int, 
                 learning_rate: float, discount_rate=0.5, fps=60, show_gui=False) -> None:
        '''
        Environment for the neural net to interact with the game and train itself.

        :param input_size: (int) Amount of input parameters the net accepts
        :param hidden_size: (list[int]) The ith element describes the amount of neurons the ith hidden layer contains.
        :param output_size: (int) Amount of output parameters returned.
        :param learning_rate: (float) How steeply gradient descent acts.
        :param discount_rate: (float) How much to value future steps (needed for sum convergence).
        :param fps: (int) GUI framerate.
        :param show_gui: (bool) If True, shows the GUI. Set to False to train quickly.
        '''
        if discount_rate <= 0 or discount_rate > 1:
            raise ValueError('discount_rate must be within (0,1]')

        self.net = LinearQNet(input_size, hidden_size, output_size)
        self.game = SnakeGameAI(fps=fps, show_gui=show_gui)

        self.discount_rate = discount_rate

        # Using the Adam optimiser with a mean-squared error loss function.
        self.optimiser = optim.Adam(self.net.parameters(), lr=learning_rate)
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