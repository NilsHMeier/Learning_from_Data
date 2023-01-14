from Neural_Networks import DenseNetwork
import numpy as np


def task_1():
    print('- - - Task 1 - - -')
    # Design a network that implements the following function: A AND ~B
    print('Network: A AND ~B')
    network_1 = DenseNetwork(input_features=2, layers=[1], activation='sign', weights=[np.array([[1, -1]])],
                             biases=[np.array([-1.5])], loss='mse')
    features = np.array([[-1, -1], [1, -1], [-1, 1], [1, 1]])
    for f in features:
        print(f'Input: {f}, Output: {network_1.predict(f)}')

    # Design a network that implements the following function: A XOR B
    print('Network: A XOR B')
    network_2 = DenseNetwork(input_features=2, layers=[2, 1], activation='sign', loss='mse',
                             weights=[np.array([[1, 1], [1, 1]]), np.array([[1, -1]])],
                             biases=[np.array([1.5, -1.5]), np.array([0])])
    for f in features:
        print(f'Input: {f}, Output: {network_2.predict(f)}')


def task_2():
    print('- - - Task 2 - - -')
    # Create the network from the task
    network = DenseNetwork(input_features=2, layers=[2, 1], activation='sigmoid', loss='mse',
                           weights=[np.array([[0.5, 0.25], [1.0, 2.0]]), np.array([[0.4, 0.5]])],
                           biases=[np.array([0, 0]), np.array([0])])

    # Calculate the output of the network for the input [0.1, 0.2]
    print(network.predict(np.array([0.1, 0.2])))

    # Calculate the gradients and the deltas for the input [0.1, 0.2] and the label 1.0
    gradients, deltas = network.get_gradients(np.array([0.1, 0.2]), np.array([1.0]))
    for i, g in enumerate(gradients):
        print(f'Gradients for layer {i}: {g}')
    for i, d in enumerate(deltas):
        print(f'Deltas for layer {i}: {d}')


def train_networks():
    print('- - - Train Networks - - -')
    # Build a network and train it to learn the AND function
    and_network = DenseNetwork(input_features=2, layers=[1], activation='tanh', loss='mse', learning_rate=0.05)
    features = np.array([[-1, -1], [1, -1], [-1, 1], [1, 1]])
    labels = np.array([-1, -1, -1, 1])
    and_network.train(features, labels, epochs=1000)

    # Predict the features and evaluate the loss
    for f, l in zip(features, labels):
        print(f'Input: {f}, Output: {and_network.predict(f)}, Label: {l}')
    print(f'Loss: {and_network.evaluate(features, labels)}')


def main():
    task_1()
    print('\n')
    task_2()
    print('\n')
    train_networks()


if __name__ == '__main__':
    main()
