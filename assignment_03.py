import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from shapely.geometry import Polygon
from sklearn import datasets
from sklearn.model_selection import train_test_split
from typing import Tuple, List, Union
from tqdm import tqdm


def make_binary_clusters(n_points=100, blob_centers: List[Tuple[float, float]] = None, cluster_std: float = 0.5,
                         linearly_separable: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generates a dataset with two classes. The classes are generated as blobs with a standard deviation of cluster_std.
    Using the linearly_separable flag, the classes can be made linearly separable.

    :param n_points: The number of data points to generate.
    :param blob_centers: The centers of the blobs. If None, the centers are randomly generated.
    :param cluster_std: The standard deviation of the blobs.
    :param linearly_separable: If True, the classes are made linearly separable.
    :return: A tuple of (data_points, labels) as NumPy arrays.
    """
    # Generate data based on the given parameters
    if blob_centers is None:
        x, y = datasets.make_blobs(n_samples=n_points, n_features=2, centers=2, random_state=0, cluster_std=cluster_std)
    else:
        x, y = datasets.make_blobs(n_samples=n_points, n_features=2, centers=blob_centers, random_state=0,
                                   cluster_std=cluster_std)

    # Check the linear separability of the data and adjust if necessary
    if linearly_separable and not is_linearly_separable(x[y == 1], x[y == 0]):
        return make_binary_clusters(n_points, blob_centers, cluster_std - 0.1, linearly_separable)
    elif not linearly_separable and is_linearly_separable(x[y == 1], x[y == 0]):
        return make_binary_clusters(n_points, blob_centers, cluster_std + 0.1, linearly_separable)

    # Return the generated data
    return x, y


def is_linearly_separable(pos_class: np.ndarray, neg_class: np.ndarray) -> bool:
    """
    This function checks if the given data points are linearly separable.

    :param pos_class: A numpy array with class 1 data points.
    :param neg_class: A numpy array with class -1 data points.
    :return: True if the data points are linearly separable, False otherwise.
    """
    pos_hull = ConvexHull(pos_class)
    neg_hull = ConvexHull(neg_class)
    return not Polygon(pos_hull.points).intersects(Polygon(neg_hull.points))


def plot_data(data_points: Union[np.ndarray, List[List[float]]], labels: Union[np.ndarray, List[int]],
              filename: str = None) -> None:
    """
    This function plots the data points.

    :param data_points: A numpy array of shape (N, D) where N is the number of data points and D is the dimensionality
    of the data points. Alternatively, a list of lists can be given.
    :param labels: A numpy array of shape (N, 1) where N is the number of data points. Each entry is either 1 or -1.
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
    for label in np.unique(labels):
        axs.scatter(data_points[labels == label, 0], data_points[labels == label, 1], marker='o',
                    label=f'Class {label}')
    axs.legend()
    axs.set(xlim=(x_min, x_max), ylim=(y_min, y_max), title='Data Points')

    # Save plot if filename is provided
    if filename is not None:
        fig.savefig(f'Figures/assignment_03_{filename}.png')
    plt.show()


def calculate_error(data_points: np.ndarray, labels: np.ndarray, weights: np.ndarray) -> float:
    """
    This function calculates the error of the given weights.

    :param data_points: A numpy array of the data points.
    :param labels: A numpy array of the labels.
    :param weights: A numpy array of the weights.
    :return: The error for the given weights.
    """
    return 1 / len(data_points) * np.sum([np.log(1 + np.exp(-y * np.dot(x, weights)))
                                          for x, y in zip(data_points, labels)])


def logistic_regression(data_points: np.ndarray, labels: np.ndarray, learning_rate: float,
                        max_iterations: int = 1000, stop_criteria: float = None) \
        -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    This function performs logistic regression on the given data points. The function returns the final weights of the
    model and the loss and weights for every training step.

    :param data_points: A numpy array of the data points.
    :param labels: A numpy array of the labels.
    :param learning_rate: The learning rate of the model.
    :param max_iterations: The maximum number of iterations.
    :param stop_criteria: The stop criteria for the model. If the loss does not change more than this value, training is
    terminated. Defaults to None.
    :return: A tuple of (weights, loss, weights_per_step) where every entry is a numpy array.
    """
    # Initialize weights
    weights = np.zeros(data_points.shape[1])

    # Set up lists for tracking the training process
    errors = []
    weights_list = [weights]

    # Start training
    prev_error = None
    for _ in range(max_iterations):
        # Calculate gradient
        gradient = np.zeros(data_points.shape[1])
        for x, y in zip(data_points, labels):
            gradient += -y * x / (1 + np.exp(y * np.dot(x, weights)))
        gradient /= len(data_points)
        # Normalize gradient
        gradient /= np.linalg.norm(gradient)
        # Update weights
        weights -= learning_rate * gradient
        # Calculate error
        error = calculate_error(data_points, labels, weights)
        errors.append(error)
        weights_list.append(weights)
        # Check if stop criteria is met
        if stop_criteria is not None:
            if prev_error is not None and abs(prev_error - error) < stop_criteria:
                break
            prev_error = error

    return weights, np.array(errors), np.array(weights_list)


def calculate_probability(x: np.ndarray, mean: np.ndarray, cov: np.ndarray) -> float:
    p = 1 / (2 * np.pi * np.linalg.det(cov) ** 0.5) * \
        np.exp(-0.5 * np.matmul(np.matmul((x - mean).T, np.linalg.inv(cov)), (x - mean)))
    return p


def gaussian_discriminant_analysis(data_points: np.ndarray, labels: np.ndarray) -> None:
    """
    This function performs Gaussian Discriminant Analysis on the given data points. The function returns the mean and
    covariance matrix of the two classes.

    :param data_points: A numpy array of the data points.
    :param labels: A numpy array of the labels.
    :return: A tuple of (mean, covariance) where every entry is a numpy array.
    """
    # Calculate mean and covariance matrix of data points
    mean = np.array([np.mean(data_points[labels == label], axis=0) for label in np.unique(labels)])
    covariance = np.array([np.cov(data_points[labels == label].T) for label in np.unique(labels)])

    # Calculate probabilities for different coordinates
    x_values = np.linspace(np.min(data_points[:, 0] - 1), np.max(data_points[:, 0] + 1), 100)
    y_values = np.linspace(np.min(data_points[:, 1] - 1), np.max(data_points[:, 1] + 1), 100)
    X, Y = np.meshgrid(x_values, y_values)
    Z_class0 = np.zeros((len(x_values), len(y_values)))
    Z_class1 = np.zeros((len(x_values), len(y_values)))
    for i, x in enumerate(x_values):
        for j, y in enumerate(y_values):
            Z_class0[j, i] = calculate_probability(np.array([x, y]), mean[0], covariance[0])
            Z_class1[j, i] = calculate_probability(np.array([x, y]), mean[1], covariance[1])

    # Plot the probabilities as a contour plot and the data points
    fig = plt.figure(figsize=(16, 8))
    axs = [fig.add_subplot(121), fig.add_subplot(122, projection='3d')]
    fig.suptitle('Gaussian Discriminant Analysis', fontsize=16)
    axs[0].contour(X, Y, Z_class0, levels=10, colors='blue', label='Class 0')
    axs[0].contour(X, Y, Z_class1, levels=10, colors='red', label='Class 1')
    for label in np.unique(labels):
        axs[0].scatter(data_points[labels == label, 0], data_points[labels == label, 1], marker='o',
                       label=f'Class {label}')
    axs[0].legend()
    axs[0].set(xlim=(x_values[0], x_values[-1]), ylim=(y_values[0], y_values[-1]),
               title='Contour plot of the probabilities', xlabel='x', ylabel='y')
    axs[1].plot_surface(X, Y, np.maximum(Z_class0, Z_class1), cmap='coolwarm')
    axs[1].set(xlabel='x', ylabel='y', title='Surface Plot of the probabilities')
    plt.tight_layout()
    fig.savefig('Figures/assignment_03_gda.png')
    plt.show()


def main():
    # Generate data and plot it
    data_points, labels = make_binary_clusters(n_points=200, cluster_std=1.0, linearly_separable=False)
    plot_data(data_points, labels, filename='data')

    # Split data into training and test set
    X_train, X_test, y_train, y_test = train_test_split(data_points, labels, test_size=0.3, random_state=42)

    # Perform logistic regression
    weights, errors, weights_per_step = logistic_regression(data_points=X_train, labels=y_train, learning_rate=1.0,
                                                            max_iterations=1000, stop_criteria=0.001)
    print(f'Final weights: {weights}')
    print(f'Final error: {errors[-1]}')
    print(f'Number of training steps: {len(errors)}')

    # Calculate the loss on the train set and test set
    train_loss = calculate_error(X_train, y_train, weights)
    print(f'Train loss: {train_loss}')
    test_loss = calculate_error(X_test, y_test, weights)
    print(f'Test loss: {test_loss}')

    # Experiment with different learning rates
    learning_rates = np.arange(0.01, 10, 0.01)
    errors = []
    for lr in tqdm(learning_rates):
        weights, _, _ = logistic_regression(data_points=X_train, labels=y_train, learning_rate=lr, stop_criteria=0.001)
        errors.append(calculate_error(X_test, y_test, weights))

    # Plot the loss for different learning rates
    fig, axs = plt.subplots()
    axs.plot(learning_rates, errors)
    axs.set(xlabel='Learning rate', ylabel='Loss', xscale='log', title='Loss for different learning rates')
    fig.savefig('Figures/assignment_03_learning_rates.png')
    plt.show()

    # Perform Gaussian Discriminant Analysis
    gaussian_discriminant_analysis(data_points=X_train, labels=y_train)


if __name__ == '__main__':
    main()
