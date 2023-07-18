import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
import os

from Models.neural_net import *
from Games.snake_game import SnakeGameAI
from matplotlib import pyplot as plt

class Environment():
    def __init__(self, model_name=None, input_size=11, hidden_size=[250], output_size=3, epochs=100,
                 batch_size=1000, learning_rate=0.001, discount_rate=0.9, epsilon_decay_rate=0.98,
                  fps=20, show_gui=False, loaded_model=False) -> None:
        '''
        Environment for the neural net to interact with the game and train itself.

        :param model_name: (str) Name for the .pth file
        :param input_size: (int) Amount of input parameters the net accepts.
        :param hidden_size: (list[int]) The ith element describes the amount of neurons the ith hidden layer contains.
        :param output_size: (int) Amount of output parameters returned.
        :param epochs: (int) Amount of games to play.
        :param batch_size: (int) Amount of states to use for training after a game completes.
        :param learning_rate: (float) How steeply gradient descent acts.
        :param discount_rate: (float) How much to value future steps (needed for sum convergence).
        :param epsilon_decay_rate: (float) epsilon = epsilon*(eps_decay_rate^N), where N is the amount of games played.
        :param fps: (int) GUI framerate.
        :param show_gui: (bool) If True, shows the GUI. Set to False to train quickly.
        :param loaded_model: (bool) If True, use a pre-trained model rather than actively training one now.
        '''
        if discount_rate <= 0 or discount_rate > 1:
            raise ValueError('discount_rate must be within (0,1].')
        if epsilon_decay_rate <= 0 or epsilon_decay_rate > 1:
            raise ValueError('epsilon_decay_rate must be within (0,1].')
        if model_name is None and not loaded_model:
            raise ValueError("If a pretrained model isn't being loaded, a name must be specified.")

        # Defining parameters
        self.model_name = model_name
        self.loaded_model = loaded_model
        self.epochs = epochs
        self.batch_size = batch_size
        self.output_size = output_size
        self.discount_rate = discount_rate
        self.epsilon = 1
        self.epsilon_decay_rate = epsilon_decay_rate

        self.memory = []

        # Generating the net and the game to be played
        self.net = LinearQNet(input_size, hidden_size, output_size)
        self.game = SnakeGameAI(fps=fps, show_gui=show_gui)

        # Using the Adam optimiser with a mean-squared error loss function.
        self.optimiser = optim.Adam(self.net.parameters(), lr=learning_rate)
        self.loss_function = nn.MSELoss()

    def train_step(self, states, actions, next_states, rewards, game_overs):
        states = torch.tensor(np.array(states), dtype=torch.float)
        actions = torch.tensor(np.array(actions), dtype=torch.float)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float)
        rewards = torch.tensor(np.array(rewards), dtype=torch.float)

        # If there's only one parameter (i.e. short term memory), we need to turn it into a tuple/unsqueezed tensor to retain the required shape
        if len(states.shape) == 1:
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

        # With probability 1 - epsilon, we use the net for a move
        if self.loaded_model or random.random() <= 1 - self.epsilon:
            prediction = self.net(torch.tensor(state, dtype=torch.float))
            idx = torch.argmax(prediction).item()
            action[idx] = 1
            
        # Otherwise, we take a random action (i.e. pick a random index in the action vector)
        else:
            idx = random.randint(0, self.output_size - 1)
            action[idx] = 1

        return action
    
    def train_batch(self):
        if len(self.memory) > self.batch_size:
            random_sample = random.sample(self.memory, self.batch_size)
        else:
            random_sample = self.memory
        
        states, actions, next_states, rewards, game_overs = zip(*random_sample)
        self.train_step(states, actions, next_states, rewards, game_overs)

    def run_training(self):
        scores = []
        epochs = range(self.epochs)
        for epoch in epochs:
            # The net plays a game
            game_over = False
            while not game_over:
                state = self.game.get_state()
                action = self.get_action(state)
                game_over, _, reward = self.game.play_step(action)
                next_state = self.game.get_state()

                # Training on a single timestep
                self.train_step(state, action, next_state, reward, game_over)

                self.memory.append((state, action, next_state, reward, game_over))

            # When a game ends, train the net on the entire available dataset (all prior games).
            scores.append(self.game.score)
            print(f"Epoch: {epoch}, Score: {self.game.score}")
            self.game.reset()
            self.epsilon *= self.epsilon_decay_rate # Updating epsilon
            self.train_batch()

        save_path = os.path.join("Outputs", "Trained Models", f"{self.model_name}.pth")
        torch.save(self.net.state_dict(), save_path)
        print(f"Trained model {self.model_name} and saved to PATH: {save_path}")

    def play_trained_model(self, model_path: os.PathLike):
        self.net.load_state_dict(torch.load(model_path))

        game_over = False
        while not game_over:
            state = self.game.get_state()
            action = self.get_action(state)
            game_over, _, _ = self.game.play_step(action)
            

if __name__=="__main__":
    # env = Environment(model_name="Mark", input_size=11, hidden_size=[250], output_size=3, epochs=200,
    #                   batch_size=1000, learning_rate=0.002, discount_rate=0.9,
    #                   epsilon_decay_rate=0.98)
    # env.run_training()

    env_trained = Environment(fps=20, show_gui=True, loaded_model=True)
    env_trained.play_trained_model(os.path.join("Outputs", "Trained Models", "Mark.pth"))