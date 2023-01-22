import itertools
import time
import multiprocessing as mp
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from tqdm import tqdm
from Models import RidgeRegression, run_experiment_for_settings

# Set up parameters
USE_THREADING = True
N_THREADS = 4


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
        # Fit a model
        model = RidgeRegression(gamma=gamma, delta=d).fit(features=x_values, labels=y_values)

        # Make prediction for every x value
        y_pred = model.predict(x_ticks)

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
    experiment_settings = [{'delta': delta, 'gamma': gamma, 'n_points': n_points, 'n_experiments': n_experiments,
                            'lower': lower, 'upper': upper}
                           for delta, gamma in itertools.product(delta_values, gamma_values)]

    # Measure the time
    start = time.time()

    # Run the experiments
    print('Running experiments...')
    np.random.seed(0)

    # Run the experiments in parallel or sequentially
    if USE_THREADING:
        with mp.Pool(processes=N_THREADS) as pool:
            results = pool.map(run_experiment_for_settings, experiment_settings)

        results = np.array(results).reshape(len(delta_values), len(gamma_values))
    else:
        # Create a matrix to store the results
        results = np.zeros((len(delta_values), len(gamma_values)))

        # Run the experiments
        for settings in tqdm(experiment_settings):
            # Store the mean of the mean squared errors
            i, j = delta_values.index(settings['delta']), gamma_values.index(settings['gamma'])
            results[i, j] = run_experiment_for_settings(settings=settings)

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
