import argparse

from environment import Environment

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model_name', type=str, help='Name of the .pth once the model trains.')
    parser.add_argument('input_size', type=int, help='Dimension of the input tensor.')
    parser.add_argument('hidden_shape', type=list, help='List of ints specifying the hidden layer neuron counts.')
    parser.add_argument('output_size', type=int, help='Dimension of the output tensor.')
    parser.add_argument('epochs', type=int, default=100, help='Amount of games to play before training ends.')
    parser.add_argument('batch_size', type=int, default=1000, help='Amount of training states to run backprop on at a time.')
    parser.add_argument('learning_rate', type=float, default=0.001, help='Rate of gradient descent.')
    parser.add_argument('discount_rate', type=float, default=0.9, help='Extent to which future states should be valued (mainly just a trick to let the return sum converge)')
    parser.add_argument('epsilon_decay_rate', type=float, default=0.98, help='epsilon(N) = epsilon(0)*(decay_rate^N), where N is the amount of completed epochs/games.')
    args = parser.parse_args()

    env = Environment(args.model_name, args.input_size, args.hidden_shape, args.output_size, args.epochs,
                      args.batch_size, args.learning_rate, args.discount_rate, args.epsiolon_decay_rate)
    env.run_training()