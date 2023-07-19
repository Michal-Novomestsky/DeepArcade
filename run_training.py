import argparse

from environment import Environment

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, help='Name of the .pth once the model trains.')
    parser.add_argument('--input_size', type=int, help='Dimension of the input tensor.')
    parser.add_argument('--hidden_shape', nargs='+', type=int, help='List of ints specifying the hidden layer neuron counts.')
    parser.add_argument('--output_size', type=int, help='Dimension of the output tensor.')
    parser.add_argument('--episodes', type=int, default=100, help='Amount of games to play before training ends.')
    parser.add_argument('--batch_size', type=int, default=32, help='Amount of training states to run backprop on at a time.')
    parser.add_argument('--max_memory', type=int, default=1000, help='Maximum amount of states stoarable in replay memory.')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Rate of gradient descent.')
    parser.add_argument('--discount_rate', type=float, default=0.9, help='Extent to which future states should be valued (mainly just a trick to let the return sum converge)')
    parser.add_argument('--epsilon_decay_rate', type=float, default=75, help='Amount of games/episodes to run before the min epsilon is reached.')
    parser.add_argument('--min_epsilon', type=float, default=0.1, help='The floor value for epsilon which remains constant after it is hit.')
    parser.add_argument('--multiprocessing', action='store_true', default=False, help='If True, enables parallelisation. Logging only available if this is False.')
    parser.add_argument('--cpu_fraction', type=float, default=1, help='% of available threads to use for multiprocessing.')
    args = parser.parse_args()

    env = Environment(args.model_name, args.input_size, args.hidden_shape, args.output_size, args.episodes,
                      args.batch_size, args.max_memory, args.learning_rate, args.discount_rate, 
                      args.epsilon_decay_rate, args.min_epsilon, args.multiprocessing, args.cpu_fraction)
    env.run_training()