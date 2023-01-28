from Neural_Networks import DenseNetwork
import numpy as np
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt


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


def task_4(constant: float = None, title: str = None):
    print(f'Initialize the weights with a constant of {constant}')

    # Create a binary classification dataset
    features, labels = make_classification(n_samples=100, n_features=5, n_redundant=0, n_informative=2,
                                           n_clusters_per_class=1, random_state=1)

    # Create a plot to visualize the network before and after training
    fig, axs = plt.subplots(ncols=2, figsize=(16, 6), tight_layout=True)

    # Create a network
    network = DenseNetwork(input_features=features.shape[1], layers=[1], activation='sigmoid', loss='mse',
                           learning_rate=0.05, init_constant=constant)

    # Plot and evaluate the network before training
    network.plot_network(ax=axs[0])
    loss = network.evaluate(features, labels)
    axs[0].set_title(f'Before Training: Loss = {loss:.2f}')

    # Train the network
    network.train(features, labels, epochs=50)

    # Plot and evaluate the network after training
    network.plot_network(ax=axs[1])
    loss = network.evaluate(features, labels)
    axs[1].set_title(f'After Training: Loss = {loss:.2f}')

    # Show the plot
    fig.suptitle(f'Constant: {constant}', fontsize=16)
    fig.show()

    # Save the plot if a title is given
    if title is not None:
        fig.savefig(f'Figures/assignment_09_{title}.png')


def main():
    task_1()
    print('\n')
    task_2()
    print('\n')
    train_networks()
    print('\n')
    task_4(constant=0.0, title='Init_0')
    task_4(constant=1.0, title='Init_1')


if __name__ == '__main__':
    main()
