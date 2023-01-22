import itertools
import time
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import mean_squared_error
from tqdm import tqdm


def determine_weights_ridge(x_values: np.ndarray, y_values: np.ndarray, gamma: float, delta: float) -> np.ndarray:
    """
    Determine the weights for the given x and y values using the kernel trick. The weights are calculated using the
    pseudo inverse of the kernel matrix. The kernel matrix is calculated using the RBF kernel.

    :param x_values: x values as a numpy array.
    :param y_values: y values as a numpy array.
    :param gamma: Gamma value for the RBF kernel.
    :param delta: Lambda value for the ridge regression.
    :return: Weights as a numpy array.
    """
    # Build Phi matrix
    meshgrid = np.meshgrid(x_values, x_values)
    phi = np.exp(-gamma * (meshgrid[1] - meshgrid[0]) ** 2)

    # Calculate the weights
    inverse = np.linalg.inv(np.dot(phi.T, phi) + (delta ** 2 * np.eye(len(phi))))
    w = np.dot(np.dot(inverse, phi.T), y_values)
    return w


def run_experiment_for_settings(n_points: int, n_experiments: int, lower: float, upper: float, gamma: float,
                                delta: float) -> float:
    """
    Run multiple experiments for the given settings and return the average MSE.

    :param n_points: Number of points to sample.
    :param n_experiments: Number of experiments to run.
    :param lower: Lower bound of the x values.
    :param upper: Upper bound of the x values.
    :param gamma: Gamma value for the RBF kernel.
    :param delta: Lambda value for the ridge regression.
    :return: Mean of the MSE values.
    """
    # Run the experiment multiple times
    mse = []
    successful_runs = 0
    while successful_runs < n_experiments:
        try:
            # Sample x values and calculate the respective y with noise
            x_values = np.random.uniform(low=lower, high=upper, size=n_points)
            y_values = x_values ** 2 + np.random.normal(loc=0, scale=0.1, size=n_points)

            # Calculate the weights
            w = determine_weights_ridge(x_values=x_values, y_values=y_values, gamma=gamma, delta=delta)

            # Make prediction for every x values between -1 and 1
            x_ticks = np.linspace(lower, upper, 200)
            y_pred = np.array([np.sum(w * np.exp(-gamma * (x_values - new_x) ** 2)) for new_x in x_ticks])

            # Calculate the mean squared error
            mse.append(mean_squared_error(y_true=x_ticks ** 2, y_pred=y_pred))
            successful_runs += 1
        except np.linalg.LinAlgError:
            pass

    return np.mean(mse)


def delta_experiments(n_points: int = 20, lower: int = -1, upper: int = 1, gamma: float = 1.0):
    """
    Fit models using a fixed gamma value and study the impact of delta to the expected generalization of the model.

    :param n_points: Number of points to sample.
    :param lower: Lower bound of the x values.
    :param upper: Upper bound of the x values.
    :param gamma: Gamma value for the RBF kernel.
    :return: None.
    """
    # Sample x values and calculate the respective y with noise
    np.random.seed(0)
    x_values = np.random.uniform(low=lower, high=upper, size=n_points)
    x_ticks = np.linspace(min(x_values), max(x_values), 100)
    y_values = x_values ** 2 + np.random.normal(loc=0, scale=0.1, size=n_points)

    # Create plot for the different delta values
    fig, axs = plt.subplots(3, 2, tight_layout=True, figsize=(12, 8))
    axs = axs.flatten()

    # Use different values for delta to fit the data
    for d, ax in zip([0, 0.01, 0.1, 1, 10, 100], axs):
        # Get the weights
        w = determine_weights_ridge(x_values=x_values, y_values=y_values, gamma=gamma, delta=d)

        # Make prediction for every x values between -1 and 1
        y_pred = np.array([np.sum(w * np.exp(-gamma * (x_values - new_x) ** 2)) for new_x in x_ticks])

        # Plot the predictions
        ax.scatter(x_values, y_values, label='Data Points')
        ax.plot(x_ticks, y_pred, label=f'Delta={d}')
        ax.legend()
        ax.set(title=f'Delta={d}', xlabel='x', ylabel='y', xlim=(lower, upper))

    # Show the plot
    fig.savefig('Figures/assignment_11_Deltas.png')
    plt.show()


def combined_experiments(n_points: int = 20, n_experiments: int = 100, lower: int = -1, upper: int = 1):
    """
    Fit models using different gamma and delta values and study the impact of the parameters to the expected
    generalization of the model. The experiments are run multiple times and the median of the mean squared errors is
    used to compare the different settings.

    :param n_points: Number of points to sample.
    :param n_experiments: Number of experiments to run.
    :param lower: Lower bound of the x values.
    :param upper: Upper bound of the x values.
    :return: None.
    """
    # Set up the different values for delta and gamma
    delta_values = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10]
    gamma_values = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10]
    experiment_settings = list(itertools.product(delta_values, gamma_values))

    # Create a matrix to store the results
    results = np.zeros((len(delta_values), len(gamma_values)))

    # Measure the time
    start = time.time()

    # Run the experiments
    print('Running experiments...')
    np.random.seed(0)
    for delta, gamma in tqdm(experiment_settings):
        # Store the mean of the mean squared errors
        i, j = delta_values.index(delta), gamma_values.index(gamma)
        results[i, j] = run_experiment_for_settings(n_points=n_points, n_experiments=n_experiments, lower=lower,
                                                    upper=upper, gamma=gamma, delta=delta)

    # Measure the time
    end = time.time()
    print(f'Finished in {round(end - start, 2)} seconds.')

    # Plot the results as a heatmap
    fig, ax = plt.subplots(figsize=(12, 8), tight_layout=True)
    sns.heatmap(results, annot=True, ax=ax, xticklabels=gamma_values, yticklabels=delta_values, cmap='viridis')
    ax.set(title='Mean Squared Error', xlabel='Gamma', ylabel='Delta')
    fig.savefig('Figures/assignment_11_Combined.png')
    plt.show()


if __name__ == '__main__':
    # Run the experiments for the different delta values
    delta_experiments(gamma=0.1, lower=-1, upper=1, n_points=20)

    # Run the experiments for the different delta and gamma values
    combined_experiments(n_points=20, n_experiments=100, lower=-1, upper=1)
