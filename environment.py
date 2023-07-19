import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
import random
import numpy as np
import os
import pandas as pd
import time

from Models.neural_net import *
from Games.snake_game import SnakeGameAI

def write_message(message, filename: os.PathLike, writemode='a'):
    """
    Writes a message to both the terminal and the output file at dir.

    :param message: Message to print. Can be any printable datatype.
    :param dir: (os.PathLike) Path to the output file.
    :param writemode: (str) Default IO write method (e.g. 'a' for append, 'w' for write, etc.).
    """
    dir = os.path.join(os.getcwd(), filename)

    print(message)
    with open(dir, writemode) as f:
        f.write(message + '\n')

class Environment():
    def __init__(self, model_name=None, input_size=11, hidden_shape=[250], output_size=3, epochs=100,
                 batch_size=1000, learning_rate=0.001, discount_rate=0.9, epsilon_decay_rate=0.98,
                  multiprocessing=False, cpu_fraction=1, show_gui=False, fps=20, loaded_model=False) -> None:
        '''
        Environment for the neural net to interact with the game and train itself.

        :param model_name: (str) Name for the .pth file
        :param input_size: (int) Amount of input parameters the net accepts.
        :param hidden_shape: (list[int]) The ith element describes the amount of neurons the ith hidden layer contains.
        :param output_size: (int) Amount of output parameters returned.
        :param epochs: (int) Amount of games to play.
        :param batch_size: (int) Amount of states to use for training after a game completes.
        :param learning_rate: (float) How steeply gradient descent acts.
        :param discount_rate: (float) How much to value future steps (needed for sum convergence).
        :param epsilon_decay_rate: (float) epsilon = epsilon*(eps_decay_rate^N), where N is the amount of games played.
        :param multiprocessing: (bool) If True, enables parallelisation. Logging only available if this is False.
        :param cpu_fraction: (float) % of available threads to use for multiprocessing.
        :param show_gui: (bool) If True, shows the GUI. Set to False to train quickly.
        :param fps: (int) GUI framerate.
        :param loaded_model: (bool) If True, use a pre-trained model rather than actively training one now.
        '''
        if discount_rate <= 0 or discount_rate > 1:
            raise ValueError('discount_rate must be within (0,1].')
        if epsilon_decay_rate <= 0 or epsilon_decay_rate > 1:
            raise ValueError('epsilon_decay_rate must be within (0,1].')
        if cpu_fraction <= 0 or cpu_fraction > 1:
            raise ValueError('cpu_fraction must be within (0,1].')
        if model_name is None and not loaded_model:
            raise ValueError("If a pretrained model isn't being loaded, a name must be specified.")

        # Defining parameters
        self.model_name = model_name
        self.loaded_model = loaded_model
        self.hidden_shape = hidden_shape
        self.output_size = output_size
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.discount_rate = discount_rate
        self.epsilon = 1
        self.epsilon_decay_rate = epsilon_decay_rate
        self.multiprocessing = multiprocessing

        if self.multiprocessing:
            self.core_count = int(np.ceil(cpu_fraction*mp.cpu_count()))
            write_message(f"Using {100*cpu_fraction}% of available cores -> {self.core_count}/{mp.cpu_count()}", filename='training_log.txt', writemode='w')
        else:
            write_message(f"Multiprocessing disabled, logging enabled.", filename='training_log.txt', writemode='w')

        self.memory = []

        # Enabling GPU functionality if available
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            write_message(f'CUDA available. Running on device/s:', filename='training_log.txt')
            for core in range(torch.cuda.device_count()):
                write_message(f'{torch.cuda.get_device_name(core)}', filename='training_log.txt')
        else:
            self.device = torch.device('cpu')
            write_message('CUDA unavailable. Switing to CPU instead.', filename='training_log.txt')

        # Generating the net and the game to be played
        self.net = LinearQNet(input_size, hidden_shape, output_size, self.device)
        self.game = SnakeGameAI(fps=fps, show_gui=show_gui)

        # Using the Adam optimiser with a mean-squared error loss function.
        self.optimiser = optim.Adam(self.net.parameters(), lr=self.learning_rate)
        self.loss_function = nn.MSELoss()

    def train_step(self, states, actions, next_states, rewards, game_overs):
        '''
        Run backpropagation on a batch of arbitrary size.

        :param states: (arrayLike) An array of states (arbitrary and depends on game).
        :param actions: (arrayLike) An array of actions to take given the state.
        :param next_states: (arrayLike) An array of subsequent states after action is taken.
        :param rewards: (arrayLike) An array of rewards for the actions taken.
        :param game_overs: (arrayLike) An array of game over flags (True if the game has ended in next_state).
        '''
        states = torch.tensor(np.array(states), dtype=torch.float).to(self.device)
        actions = torch.tensor(np.array(actions), dtype=torch.float).to(self.device)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float).to(self.device)
        rewards = torch.tensor(np.array(rewards), dtype=torch.float).to(self.device)

        # If there's only one parameter (i.e. short term memory), we need to turn it into a tuple/unsqueezed tensor to retain the required shape
        if len(states.shape) == 1:
            states = torch.unsqueeze(states,0).to(self.device)
            next_states = torch.unsqueeze(next_states,0).to(self.device)
            actions = torch.unsqueeze(actions,0).to(self.device)
            rewards = torch.unsqueeze(rewards,0).to(self.device)
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

        :param state: (np.ndarray) A numpy array describing the gamestate. Arbitrary and depends on the game played.
        :return: (np.ndarray) The action the agent takes (game-dependent).
        '''
        action = np.zeros(self.output_size, dtype=np.float64)

        # With probability 1 - epsilon, we use the net for a move
        if self.loaded_model or random.random() <= 1 - self.epsilon:
            prediction = self.net(torch.tensor(state, dtype=torch.float).to(self.device))
            idx = torch.argmax(prediction).item()
            action[idx] = 1
            
        # Otherwise, we take a random action (i.e. pick a random index in the action vector)
        else:
            idx = random.randint(0, self.output_size - 1)
            action[idx] = 1

        return action
    
    def train_batch(self) -> None:
        '''
        Train on a random sample of data of size self.batch_size across all played games.
        '''
        if len(self.memory) > self.batch_size:
            random_sample = random.sample(self.memory, self.batch_size)
        else:
            random_sample = self.memory
        
        states, actions, next_states, rewards, game_overs = zip(*random_sample)
        self.train_step(states, actions, next_states, rewards, game_overs)

    def training_process(self, epochs, multiprocessing=False):
        if not multiprocessing:
            epsilon_log = []
            return_log = []
            time_log = []

        for _ in range(epochs):
            t0 = time.perf_counter()
            game_over = False
            while not game_over:
                state = self.game.get_state()
                action = self.get_action(state)
                game_over, reward = self.game.play_step(action)
                next_state = self.game.get_state()

                # Training on a single timestep
                self.train_step(state, action, next_state, reward, game_over)

                self.memory.append((state, action, next_state, reward, game_over))

            # Resetting for a new epoch
            self.game.reset()
            self.train_batch() # When a game ends, train the net on a subset of all available data (all prior games).
            t1 = time.perf_counter()

            # Logging stats
            if not multiprocessing:
                return_log.append(self.game.score)
                epsilon_log.append(self.epsilon)
                time_log.append(t1-t0)

            self.epsilon *= self.epsilon_decay_rate # Updating epsilon 

        if not multiprocessing:
            return return_log, epsilon_log, time_log

    def run_training(self):
        '''
        Trains the net using the parameters set.
        '''

        write_message("Training started.", filename='training_log.txt')

        t0 = time.perf_counter()
        processes = []

        if self.multiprocessing:
            for _ in range(self.core_count):
                p = mp.Process(target=self.training_process, args=(self.epochs//self.core_count, True))
                p.start()
                processes.append(p)
            for p in processes:
                p.join()

        else:
            return_log, epsilon_log, time_log = self.training_process(self.epochs)

        t1 = time.perf_counter()

        save_path = os.path.join("Outputs", "Trained Models", f"{self.model_name}.pth")
        torch.save(self.net.state_dict(), save_path)
        write_message(f"Trained model {self.model_name} in {round((t1-t0)/60, 1)}min. Saved to PATH: {save_path}", filename='training_log.txt')

        if not self.multiprocessing:
            save_path = os.path.join("Outputs", "Logs", f"{self.model_name}.csv")
            df = pd.DataFrame({'return': return_log, 'epsilon': epsilon_log, 'time': time_log,
                            'batch_size': [self.batch_size]*self.epochs, 'learning_rate': [self.learning_rate]*self.epochs,
                            'discount_rate': [self.discount_rate]*self.epochs, 'epsilon_decay_rate': [self.epsilon_decay_rate]*self.epochs,
                            'hidden_layers': [len(self.hidden_shape)]*self.epochs, 'mean_layer_width': [np.mean(self.hidden_shape)]*self.epochs})
            df.to_csv(save_path)
            write_message(f"Log written to PATH: {save_path}", filename='training_log.txt')

    def play_trained_model(self, model_path: os.PathLike) -> None:
        '''
        Run a game with a pretrained model.

        :param model_path: (os.PathLike) Path to the trained model.
        '''
        self.net.load_state_dict(torch.load(model_path))
        self.net.eval()

        game_over = False
        while not game_over:
            state = self.game.get_state()
            action = self.get_action(state)
            game_over, _ = self.game.play_step(action)

        print(f'Final Score: {self.game.score}')
            
if __name__=="__main__":
    ### Run a trained model ###
    env_trained = Environment(fps=20, show_gui=True, loaded_model=True)
    env_trained.play_trained_model(os.path.join("Outputs", "Trained Models", "1_layer_200_epoch_testrun.pth"))