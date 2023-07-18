import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np

from Models.neural_net import *
from Games.snake_game import SnakeGameAI
from collections import deque

class Environment():
    def __init__(self, input_size: int, hidden_size: list, output_size: int, batch_size=1000, 
                 learning_rate=0.001, discount_rate=0.9, epsilon_decay_rate=0.98,
                  fps=20, show_gui=False) -> None:
        '''
        Environment for the neural net to interact with the game and train itself.

        :param input_size: (int) Amount of input parameters the net accepts
        :param hidden_size: (list[int]) The ith element describes the amount of neurons the ith hidden layer contains.
        :param output_size: (int) Amount of output parameters returned.
        :param batch_size: (int) Amount of states to use for training after a game completes.
        :param learning_rate: (float) How steeply gradient descent acts.
        :param discount_rate: (float) How much to value future steps (needed for sum convergence).
        :param epsilon_decay_rate: (float) epsilon = epsilon*(eps_decay_rate^N), where N is the amount of games played.
        :param fps: (int) GUI framerate.
        :param show_gui: (bool) If True, shows the GUI. Set to False to train quickly.
        '''
        if discount_rate <= 0 or discount_rate > 1:
            raise ValueError('discount_rate must be within (0,1]')
        if epsilon_decay_rate <= 0 or epsilon_decay_rate > 1:
            raise ValueError('epsilon_decay_rate must be within (0,1]')

        # Defining parameters
        self.games_played = 0
        self.batch_size = batch_size
        self.output_size = output_size
        self.discount_rate = discount_rate
        self.epsilon = 1
        self.epsilon_decay_rate = epsilon_decay_rate

        self.memory = deque()

        # Generating the net and the game to be played
        self.net = LinearQNet(input_size, hidden_size, output_size)
        self.game = SnakeGameAI(fps=fps, show_gui=show_gui)

        # Using the Adam optimiser with a mean-squared error loss function.
        self.optimiser = optim.Adam(self.net.parameters(), lr=learning_rate)
        self.loss_function = nn.MSELoss()

    def train_step(self, states, actions, next_states, rewards, game_overs):
        states = torch.tensor(states, dtype=torch.float)
        actions = torch.tensor(actions, dtype=torch.float)
        next_states = torch.tensor(next_states, dtype=torch.float)
        rewards = torch.tensor(rewards, dtype=torch.float)

        if len(states.shape) == 1: # only one parameter to train , Hence convert to tuple of shape (1, x)
            #(1 , x)
            states = torch.unsqueeze(states,0)
            next_states = torch.unsqueeze(next_states,0)
            actions = torch.unsqueeze(actions,0)
            rewards = torch.unsqueeze(rewards,0)
            game_overs = (game_overs, )

        prediction = self.net(states)
        target = prediction.clone()

        for i in range(len(game_overs)):
            if game_overs[i]:
                # If there is no next state (i.e. the game ended), don't update the Q-value (since there isn't any new data to compare with).
                Q_new = rewards[i]
            else:
                # Q-value which maximises the expected reward
                Q_new = rewards[i] + self.discount_rate*torch.max(self.net(next_states[i]))

            # Updating the output node (i.e. the one that produced the Q-value used to calculate Q_new) with Q_new.
            target[i][torch.argmax(actions).item()] = Q_new
        
        # Calculating loss and running backprop.
        self.optimiser.zero_grad()
        loss = self.loss_function(target, prediction)
        loss.backward()
        self.optimiser.step()

    def get_action(self, state: np.ndarray) -> np.ndarray:
        '''
        Uses Epsilon-Greedy Action Selection to choose an action:
        Pr(random move) = epsilon,
        Pr(max(Q)) = 1 - epsilon,

        :param state: (torch.tensor) A tensor describing the gamestate. Arbitrary and depends on the game played.
        :return: (torch.tensor) The action the agent takes (game-dependent).
        '''
        action = np.zeros(self.output_size, dtype=np.float64)

        # With probability epsilon, we take a random action (i.e. pick a random index in the action vector)
        if random.random() <= self.epsilon:
            idx = random.randint(0, self.output_size - 1)
            action[idx] = 1
        # Otherwise use the net for a move
        else:
            prediction = self.net(torch.tensor(state, dtype=torch.float))
            idx = torch.argmax(prediction).item()
            action[idx] = 1

        return action
    
    def train_batch(self):
        if len(self.memory) > self.batch_size:
            random_sample = random.sample(self.memory, self.batch_size)
        else:
            random_sample = self.memory
        
        states, actions, next_states, rewards, game_overs = zip(*random_sample)
        # self.train_step(torch.tensor(states, dtype=torch.float), torch.tensor(actions, dtype=torch.float),
        #                 torch.tensor(next_states, dtype=torch.float), torch.tensor(rewards, dtype=torch.float),
        #                 game_overs)
        self.train_step(states, actions, next_states, rewards ,game_overs)


    def logger(self):
        raise NotImplementedError

    def run_training(self, epochs=1000):
        for _ in range(epochs):
            # The net plays a game
            game_over = False
            while not game_over:
                state = self.game.get_state()
                action = self.get_action(state)
                game_over, _, reward = self.game.play_step(action)
                next_state = self.game.get_state()

                self.train_step(state, action, next_state, reward, game_over)

                self.memory.append((state, action, next_state, reward, game_over))

            # When a game ends, train the net
            total_score = self.game.score
            self.game.reset()
            self.games_played += 1
            self.epsilon *= self.epsilon_decay_rate
            self.train_batch()
            print(f"Score: {total_score}")

if __name__=="__main__":
    env = Environment(input_size=11, hidden_size=[250], output_size=3,
                      batch_size=1000, learning_rate=0.001, discount_rate=0.9,
                      epsilon_decay_rate=0.98, fps=100, show_gui=True)
    
    env.run_training()


