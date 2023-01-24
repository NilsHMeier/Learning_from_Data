from .Model import BaseModel
import numpy as np


class Perceptron(BaseModel):
    """
    This class implements the perceptron model.
    """
    def __init__(self, learning_rate: float = 0.1, max_iterations: int = 1000):
        """
        Initialize the model.

        :param learning_rate: Learning rate for the perceptron.
        :param max_iterations: Maximum number of iterations.
        """
        # Check the input parameters
        assert isinstance(learning_rate, (float, int)), 'Learning rate must be a numeric.'
        assert isinstance(max_iterations, int), 'Max iterations must be an integer.'

        # Initialize the parameters
        self.converged = False
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.weights = np.zeros(0)
        self.bias = 0

    def fit(self, features: np.ndarray, labels: np.ndarray) -> 'Perceptron':
        """
        Fit the model to the given data. The weights are calculated using the perceptron algorithm.

        :param features: Features as a numpy array.
        :param labels: Labels as a numpy array.
        :return: Self.
        """
        # Initialize the weights, bias and r
        weights = np.zeros(features.shape[1])
        bias = 0
        r = max(np.linalg.norm(features, axis=1))

        # Iterate over the maximum number of iterations
        num_iterations = 0
        mistakes_made = True
        while num_iterations < self.max_iterations and mistakes_made:
            mistakes_made = False
            for i in range(features.shape[0]):
                # If the current data point is misclassified, update weights and bias
                if labels[i] * (np.dot(weights, features[i]) + bias) <= 0:
                    weights += self.learning_rate * labels[i] * features[i]
                    bias += self.learning_rate * labels[i] * r**2
                    mistakes_made = True
                    num_iterations += 1

        # Set the converged flag
        self.converged = not mistakes_made

        # Save the weights and bias
        self.weights = weights
        self.bias = bias
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        """
        Predict the targets for the given features.

        :param features: Features as a numpy array.
        :return: Predicted targets as a numpy array.
        """
        return np.sign(np.dot(features, self.weights) + self.bias)

    def evaluate(self, features: np.ndarray, labels: np.ndarray) -> float:
        """
        Evaluate the model on the given data by calculating the accuracy.

        :param features: Features as a numpy array.
        :param labels: True labels as a numpy array.
        :return: Accuracy as a float.
        """
        return np.mean(self.predict(features) == labels)
