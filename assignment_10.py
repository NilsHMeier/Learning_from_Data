import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
from typing import Tuple


def visualize_decision_boundary(weights: np.ndarray, limits: Tuple[int, int] = (-3, 3), title: str = None):
    """
    Visualize the decision boundary for a given set of weights using the feature transform z = (1, x1², x2²) and the
    hypothesis h(x) = sign(w.T * z(x)).

    :param weights: Weights of the hypothesis function.
    :param limits: Limits of the plot.
    :param title: Title of the plot.
    :return: None.
    """
    # Generate data points in x-space
    x_values = np.linspace(limits[0], limits[1], 100)
    y_values = np.linspace(limits[0], limits[1], 100)
    data_points = np.array(list(itertools.product(x_values, y_values)))

    # Transform the data points to the z-space
    z_space = np.concatenate([np.ones(shape=(len(data_points), 1)),
                              data_points ** 2],
                             axis=1)

    # Determine the labels using the weights and the sign function
    labels = np.sign(np.dot(weights, z_space.T))

    # Plot the results
    fig, ax = plt.subplots(tight_layout=True)
    sns.scatterplot(x=data_points[:, 0], y=data_points[:, 1], hue=labels, ax=ax)
    ax.set(title=f'Decision Boundary with weights {" ".join(weights.astype(str))}',
           xlim=limits, ylim=limits)
    plt.show()

    if title is not None:
        fig.savefig(f'Figures/assignment_10_{title}.png')


def determine_weights(x_values: np.ndarray, y_values: np.ndarray, gamma: float) -> np.ndarray:
    """
    Determine the weights for the given x and y values using the kernel trick. The weights are calculated using the
    pseudo inverse of the kernel matrix. The kernel matrix is calculated using the RBF kernel.

    :param x_values: x values as a numpy array.
    :param y_values: y values as a numpy array.
    :param gamma: Gamma value for the RBF kernel.
    :return: Weights as a numpy array.
    """
    # Build Phi matrix
    meshgrid = np.meshgrid(x_values, x_values)
    phi = np.exp(-gamma * (meshgrid[1] - meshgrid[0]) ** 2)

    # Calculate the weights
    w = np.dot(np.linalg.inv(phi), y_values)
    return w


def task_2(n_points: int = 20, noise_level: float = 0.05, lower: int = -1, upper: int = 1):
    """
    Generate a random dataset (x out of [lower, upper], y = x² + noise) and determine the weights for different gammas
    using the kernel trick in the above method. Based on the weights and some x values, the corresponding y values are
    calculated and plotted.

    :param n_points: Number of points to generate.
    :param noise_level: Noise level for the y values.
    :param lower: Lower limit for the x values.
    :param upper: Upper limit for the x values.
    :return: None.
    """
    # Sample x values and calculate the respective y with noise
    np.random.seed(0)
    x_values = np.random.uniform(low=lower, high=upper, size=n_points)
    x_ticks = np.linspace(min(x_values), max(x_values), 100)
    y_values = x_values ** 2 + np.random.normal(loc=0, scale=noise_level, size=n_points)

    fig, axs = plt.subplots(3, 2, tight_layout=True, figsize=(12, 8))
    axs = axs.flatten()

    # Use different values for gamma to fit the data
    for g, ax in zip([0.01, 0.05, 0.1, 0.5, 1, 5], axs):
        # Get the weights
        w = determine_weights(x_values=x_values, y_values=y_values, gamma=g)

        # Make prediction for every x values between -1 and 1
        # grids = np.meshgrid(x_values, x_ticks)
        # y_pred = np.sum(w * np.exp(-g * (grids[0] - grids[1]) ** 2), axis=1)
        y_pred = np.array([np.sum(w * np.exp(-g * (x_values - new_x) ** 2)) for new_x in x_ticks])

        # Plot the predictions
        ax.scatter(x_values, y_values, label='Data Points')
        ax.plot(x_ticks, y_pred, label=f'Gamma={g}')
        ax.legend()

    # Save the figure
    fig.savefig('Figures/assignment_10_Gammas.png')
    fig.show()


def main():
    # Task 1
    visualize_decision_boundary(weights=np.array([1, -1, -1]), title='a')
    visualize_decision_boundary(weights=np.array([-1, 1, 1]), title='b')
    visualize_decision_boundary(weights=np.array([1, -1, -2]), title='c')
    visualize_decision_boundary(weights=np.array([1, 1, -1]), title='d')

    # Task 2
    task_2(n_points=20, lower=-5, upper=5)


if __name__ == '__main__':
    main()
