import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
from typing import List, Tuple, Union


def pla_algorithm(data_points: Union[np.ndarray, List[List[float]]], labels: Union[np.ndarray, List[int]],
                  lr: float, max_iterations: int = 1000) -> Tuple[np.ndarray, float, int]:
    """
    This function implements the perceptron learning algorithm.

    :param data_points: A numpy array of shape (N, D) where N is the number of data points and D is the dimensionality
    of the data points.
    :param labels: A numpy array of shape (N, 1) where N is the number of data points. Each entry is either 1 or -1.
    :param lr: The learning rate.
    :param max_iterations: The maximum number of iterations to run the algorithm.
    :return: A numpy array of shape (D, 1) where D is the dimensionality of the data points. This is the weight vector.
    """
    # Make sure data points and labels are numpy arrays
    data_points = np.array(data_points)
    labels = np.array(labels)
    # Initialize weights to zeros, bias to 0 and r to maximum norm of data points
    weights = np.zeros(data_points.shape[1])
    b = 0
    r = max([np.linalg.norm(x) for x in data_points])
    num_iterations = 0
    mistakes_made = True
    while num_iterations < max_iterations and mistakes_made:
        mistakes_made = False
        for i in range(data_points.shape[0]):
            # If the current data point is misclassified, update weights and bias
            if labels[i] * (np.dot(weights, data_points[i]) + b) <= 0:
                weights += lr * labels[i] * data_points[i]
                b += lr * labels[i] * r**2
                mistakes_made = True
                num_iterations += 1
    return weights, b, num_iterations


def make_linearly_separable_clusters(num_points: int = 100) -> Tuple[np.ndarray, np.ndarray]:
    """
    This function generates linearly separable clusters.

    :param num_points: The number of points to generate.
    :return: A tuple of two numpy arrays. The first array is of shape (N, 2) where N is the number of data points.
    """

    x, y = datasets.make_blobs(n_samples=num_points, centers=2, n_features=2, cluster_std=1.5, center_box=(-10.0, 10.0),
                               shuffle=True, random_state=42)
    y = np.where(y == 0, -1, y)
    return x, y


def plot_data(data_points: Union[np.ndarray, List[List[float]]], labels: Union[np.ndarray, List[int]],
              weights: np.ndarray = None, bias: float = None, filename: str = None) -> None:
    """
    This function plots the data points.

    :param data_points: A numpy array of shape (N, D) where N is the number of data points and D is the dimensionality
    of the data points.
    :param labels: A numpy array of shape (N, 1) where N is the number of data points. Each entry is either 1 or -1.
    :param weights: Optional array of weights and bias to plot the decision boundary.
    :param bias: Optional bias to plot the decision boundary.
    :param filename: Optional filename to save the plot.
    """
    # Make sure data points and labels are numpy arrays
    data_points = np.array(data_points)
    labels = np.array(labels)

    # Determine the range of the data points
    x_min, x_max = data_points[:, 0].min() - 1, data_points[:, 0].max() + 1
    y_min, y_max = data_points[:, 1].min() - 1, data_points[:, 1].max() + 1

    # Plot data points
    fig, axs = plt.subplots()
    axs.scatter(data_points[labels == -1, 0], data_points[labels == -1, 1], marker='o', color='r', label='Class -1')
    axs.scatter(data_points[labels == 1, 0], data_points[labels == 1, 1], marker='o', color='g', label='Class 1')

    # Plot decision boundary
    if weights is not None:
        x = np.linspace(x_min, x_max, 2)
        y = (-weights[0] * x - bias) / weights[1]
        axs.plot(x, y, 'b--', label='Decision boundary')
    axs.legend()
    axs.set(xlim=(x_min, x_max), ylim=(y_min, y_max))

    # Save plot if filename is provided
    if filename is not None:
        fig.savefig(f'Figures/assignment_01_{filename}.png')
    plt.show()


def main():
    # Set data points and labels
    data_points = [[1, 2], [3, 2], [2, 1], [3, 3]]
    labels = [-1, 1, -1, 1]

    # Set learning rate and maximum number of iterations
    lr = 1
    max_iterations = 1000

    # Run perceptron learning algorithm
    weights, b, num_iterations = pla_algorithm(data_points, labels, lr, max_iterations)
    plot_data(data_points, labels, weights, b, 'Simple')
    quit()
    print(f'Weights: {weights}')
    print(f'Bias: {b}')
    print(f'Number of iterations: {num_iterations}')

    # Generate linearly separable clusters
    data_points, labels = make_linearly_separable_clusters()

    # Run perceptron learning algorithm
    weights, b, num_iterations = pla_algorithm(data_points, labels, lr, max_iterations)
    plot_data(data_points, labels, weights, b, 'Clusters')
    print(f'Weights: {weights}')
    print(f'Bias: {b}')
    print(f'Number of iterations: {num_iterations}')


if __name__ == '__main__':
    main()
