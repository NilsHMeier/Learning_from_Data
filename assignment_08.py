import numpy as np
import matplotlib.pyplot as plt
from cvxopt import matrix, solvers
from typing import List, Tuple


def generate_data(n: int = 20) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a toy dataset with n samples. The features are 2-dimensional where the first dimension is between 0 and 1
    and the second dimension is between 0 and 2. The labels are assigned using sign(x2).

    :param n: Number of samples. Default is 20.
    :return: A tuple of features and labels.
    """
    # Generate random data points and assign labels
    np.random.seed(0)
    x1_values = np.random.uniform(0, 1, (n, 1))
    x2_values = np.random.uniform(-1, 1, (n, 1))
    labels = np.sign(x2_values).squeeze()

    # Return features and labels
    return np.concatenate([x1_values, x2_values], axis=1), labels


def plot_dataset(features: np.ndarray, labels: np.ndarray):
    """
    Plot the given dataset.

    :param features: A 2D array of features.
    :param labels: A 1D array of labels.
    """
    # Plot the dataset
    fig, ax = plt.subplots(tight_layout=True)
    ax.scatter(features[:, 0], features[:, 1], c=labels, cmap='bwr', edgecolors='k')
    ax.set(xlabel='x1', ylabel='x2', xlim=(0, 1), ylim=(-1, 1), title='Dataset')
    fig.savefig('Figures/assignment_08_Dataset.png')
    plt.show()


def support_vector_machine(features: np.ndarray, labels: np.ndarray):
    """
    Train a support vector machine using the given features and labels.

    :param features: A 2D array of features.
    :param labels: A 1D array of labels.
    """
    # Get number of samples and features
    n_samples, n_features = features.shape

    # Create the quadratic programming problem
    P = matrix(np.outer(labels, labels) * (features @ features.T))
    q = matrix(-1 * np.ones(n_samples))
    A = matrix(labels, (1, n_samples))
    b = matrix(0.0)
    G = matrix(-1 * np.eye(n_samples))
    h = matrix(np.zeros(n_samples))

    # Solve the quadratic programming problem
    solvers.options['show_progress'] = False
    solvers.options['feastol'] = 1e-5
    solution = solvers.qp(P, q, G, h, A, b, solver='mosek')

    # Get the Lagrange multipliers
    alphas = np.array(solution['x']).squeeze()

    # Get the support vectors
    support_vector_indices = np.where(alphas > 1e-5)[0]
    support_vectors = features[support_vector_indices]

    # Calculate the weights
    weights = (alphas * labels) @ features

    # Calculate the intercept using y_n(w.Tx_n + b) = 1
    intercept = 1 / labels[support_vector_indices[0]] - np.dot(weights, support_vectors[0])

    # Return the intercept and weights
    return intercept, weights, support_vector_indices


def plot_decision_boundary(features: np.ndarray, labels: np.ndarray, intercept: float, weights: np.ndarray):
    """
    Plot the decision boundary of the given support vector machine.

    :param features: A 2D array of features.
    :param labels: A 1D array of labels.
    :param intercept: The intercept of the support vector machine.
    :param weights: The weights of the support vector machine.
    """
    # Plot the dataset
    fig, ax = plt.subplots(tight_layout=True)
    ax.scatter(features[:, 0], features[:, 1], c=labels, cmap='bwr', edgecolors='k')
    ax.set(xlabel='x1', ylabel='x2', xlim=(0, 1), ylim=(-1, 1), title='Results of SVM Algorithm')

    # Plot the margin and decision boundary
    x1_values = np.array([0, 1])
    boundary = -(intercept + weights[0] * x1_values) / weights[1]
    upper_margin = boundary + 1 / np.linalg.norm(weights)
    lower_margin = boundary - 1 / np.linalg.norm(weights)
    ax.fill_between(x1_values, upper_margin, lower_margin, color='gray', alpha=0.2, label='Margin')
    ax.plot(x1_values, boundary, 'k--', label='Decision Boundary')
    ax.legend()
    fig.savefig('Figures/assignment_08_Results.png')
    plt.show()


def main():
    features, labels = generate_data()
    plot_dataset(features, labels)

    intercept, weights, support_vector_indices = support_vector_machine(features, labels)
    print(f'{intercept=}')
    print(f'{weights=}')
    print(f'{support_vector_indices=}')
    plot_decision_boundary(features, labels, intercept, weights)


if __name__ == '__main__':
    main()
