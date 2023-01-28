from typing import List, Tuple
import numpy as np
from . import activations, losses
from .util import cast_inputs
import matplotlib.pyplot as plt
import networkx as nx


class DenseNetwork:
    """
    This class implements the basic functionalities of a dense neural network. The class can be used to create networks
    with known weights and biases, or it can be used to create a new network and train it. The class also implements the
    ability to predict the output of the network given the input features.
    The training of the network is done using the backpropagation algorithm and gradient descent.
    """

    def __init__(self, input_features: int, layers: List[int], activation: str, loss: str,
                 weights: List[np.ndarray] = None, biases: List[np.ndarray] = None,
                 learning_rate: float = 1.0):
        """
        Initialize the network. If weights and biases are not provided, they will be initialized randomly.

        :param input_features: The number of input features.
        :param layers: The number of units in each layer.
        :param activation: The activation function to use.
        :param loss: The loss function to use.
        :param weights: The optional weights of the network as a list of numpy arrays.
        :param biases: The optional biases of the network as a list of numpy arrays.
        :param learning_rate: The learning rate of the network.
        """
        # Check inputs
        assert input_features > 0, 'Input features must be greater than 0.'
        assert len(layers) > 0, 'There must be at least one layer.'
        assert layers[-1] == 1, 'The last layer must have a single output.'

        # Initialize the network
        self.__learning_rate = learning_rate
        self.__input_features = input_features
        self.__layers = layers
        self.__act = activations.get_activation_function(activation)
        self.__act_derivative = activations.get_activation_derivative(activation)
        self.__loss, self.__loss_derivative = losses.get_loss_function(loss)
        self.__weights = weights if weights is not None else self.__initialize_weights()
        self.__biases = biases if biases is not None else self.__initialize_biases()

    @property
    def weights(self):
        """
        :return: The weights of the network
        """
        return self.__weights

    @property
    def biases(self):
        """
        :return: The biases of the network
        """
        return self.__biases

    def __initialize_weights(self):
        """
        Initialize the weights of the network using the standard normal distribution.

        :return: None.
        """
        weights = []
        for i in range(len(self.__layers)):
            if i == 0:
                weights.append(np.random.randn(self.__layers[i], self.__input_features))
            else:
                weights.append(np.random.randn(self.__layers[i], self.__layers[i - 1]))
        return weights

    def __initialize_biases(self):
        """
        Initialize the biases of the network using the standard normal distribution.

        :return: None.
        """
        biases = []
        for i in range(len(self.__layers)):
            biases.append(np.random.randn(self.__layers[i]))
        return biases

    def __repr__(self):
        """
        :return: A string representation of the network.
        """
        return f'DenseNetwork(input_features={self.__input_features}, layers={self.__layers}, ' \
               f'activation={self.__act.__name__})'

    def plot_network(self, ax: plt.Axes = None) -> None:
        """
        Plot the network.

        :param ax: The axis to plot on. If None, a new figure will be created.
        :return: None.
        """
        # Create the graph and a dictionary for positions
        G = nx.Graph()
        pos = {}

        # Add the input features
        for i in range(self.__input_features):
            G.add_node(f'Input_{i}', label=f'Input Feature {i}')
            pos[f'Input_{i}'] = np.array([0, i - np.mean(range(self.__input_features))])

        # Loop over the remaining layers
        for l in range(len(self.__layers)):
            for i in range(self.__layers[l]):
                G.add_node(f'Node_{l}_{i}', label=f'Layer {l + 1}, Unit {i + 1}')
                pos[f'Node_{l}_{i}'] = np.array([0.5 * (l + 1), i - np.mean(range(self.__layers[l]))])
                if l == 0:
                    for j in range(self.__input_features):
                        G.add_edge(f'Input_{j}', f'Node_{l}_{i}', weight=round(self.weights[l][i][j], 2))
                else:
                    for j in range(self.__layers[l - 1]):
                        G.add_edge(f'Node_{l - 1}_{j}', f'Node_{l}_{i}', weight=round(self.weights[l][i][j], 2))

        # Plot the graph
        if ax is None:
            fig, ax = plt.subplots()

        nx.draw_networkx_nodes(G, pos, node_size=1000, node_color='lightblue', ax=ax)
        nx.draw_networkx_edges(G, pos, width=2, edge_color='black', ax=ax)
        nx.draw_networkx_labels(G, pos, font_size=10, font_color='black', verticalalignment='bottom',
                                horizontalalignment='center', ax=ax)
        nx.draw_networkx_edge_labels(G, pos, label_pos=0.65, font_size=8, font_color='black', ax=ax)
        plt.tight_layout()
        plt.show()

        return None

    def predict(self, features: np.ndarray) -> np.ndarray:
        """
        Predict the output of the network given the input features.

        :param features: The input features as a numpy array.
        :return: The output of the network as a numpy array.
        """
        # In case of multiple input features, call predict for each input feature
        if features.ndim == 2:
            return np.array([self.predict(f) for f in features])

        # Make sure the input features are the correct shape
        inputs = cast_inputs(features)
        assert len(inputs) == self.__input_features, 'Input features must match the number of input features.'

        # Calculate the output of the network for the given input features
        for l in range(len(self.__layers)):
            s = np.array([np.dot(inputs, self.__weights[l][i]) for i in range(self.__layers[l])]) + self.__biases[l]
            x = self.__act(s)
            inputs = x
        return inputs

    def get_gradients(self, features, label) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Calculate the gradients of the weights and biases of the network given the input features and label.

        :param features: Single input feature as a numpy array.
        :param label: The label as a numpy array.
        :return: The gradients of the weights and the deltas as a tuple of lists of numpy arrays.
        """
        # Calculate the signals and x values by forward propagation
        signals = []
        x_values = []
        inputs = cast_inputs(features)
        assert len(inputs) == self.__input_features, 'Input features must match the number of input features.'
        for l in range(len(self.__layers)):
            s = np.array([np.dot(inputs, self.__weights[l][i]) for i in range(self.__layers[l])]) + self.__biases[l]
            x = self.__act(s)
            signals.append(s)
            x_values.append(x)
            inputs = x

        # Calculate the delta for the output layer by back propagation
        delta_output = self.__loss_derivative(x_values[-1], label, signals[-1], self.__act_derivative)
        deltas = [delta_output]
        remaining_layers = list(range(len(self.__layers) - 1))[::-1]
        for l in remaining_layers:
            previous_deltas = deltas[0]
            new_deltas = []
            for unit in range(self.__layers[l]):
                new_deltas.append(sum([previous_deltas[i] *
                                       self.__weights[l + 1][i][unit] *
                                       self.__act_derivative(signals[l][unit])
                                       for i in range(len(previous_deltas))]))
            deltas.insert(0, np.array(new_deltas))

        # Use the deltas to calculate the gradients
        gradients = []
        for l in range(len(self.__layers)):
            if l == 0:
                gradients.append(np.outer(features, deltas[l]))
            else:
                gradients.append(np.outer(x_values[l - 1], deltas[l]))
        return gradients, deltas

    def apply_gradients(self, gradients: List[np.ndarray], deltas: List[np.ndarray]):
        """
        Apply the gradients to the weights and biases of the network.

        :param gradients: The gradients of the weights and biases as a list of numpy arrays.
        :param deltas: The deltas of the weights and biases as a list of numpy arrays.
        :return: None.
        """
        # Update the weights and biases
        for l in range(len(self.__layers)):
            self.__weights[l] -= gradients[l].reshape(self.__weights[l].shape) * self.__learning_rate
            self.__biases[l] -= deltas[l].reshape(self.__biases[l].shape) * self.__learning_rate

    def train(self, features: np.ndarray, labels: np.ndarray, epochs=10):
        """
        Train the network using the given features and labels.

        :param features: The input features as a numpy array.
        :param labels: The labels as a numpy array.
        :param epochs: The number of epochs to train the network for. Default is 10.
        :return: None.
        """
        # For each epoch and data point, calculate the gradients and apply them to the network
        for epoch in range(epochs):
            for i in range(len(features)):
                gradients, deltas = self.get_gradients(features[i], labels[i])
                self.apply_gradients(gradients, deltas)

    def evaluate(self, features: np.ndarray, labels: np.ndarray) -> float:
        """
        Evaluate the network by calculating the overall loss using the given features and labels.

        :param features: The input features as a numpy array.
        :param labels: The labels as a numpy array.
        :return: The overall loss as a float.
        """
        predictions = self.predict(features)
        return self.__loss(predictions, labels)
