import time
import numpy as np
import pandas as pd
from Models.Experiments import run_experiments, generate_legendre_polynomial
import matplotlib.pyplot as plt
import seaborn as sns
import multiprocessing as mp

USE_MULTIPROCESSING = True
N_THREADS = 8
N_EXPERIMENTS = 10000


def plot_legendre_polynomials(max_n: int = 5):
    # Create a figure
    fig, ax = plt.subplots(figsize=(10, 8), tight_layout=True)
    fig.suptitle('Legendre Polynomials', fontsize=16)

    # Plot the polynomials
    x = np.linspace(-1, 1, 1000)
    for i in range(max_n):
        polynomial = generate_legendre_polynomial(degree=i)
        ax.plot(x, polynomial(x), label=f'$P_{i}(x)$')

    # Add a legend
    ax.legend()

    # Save the figure
    fig.savefig('Figures/lecture_12_legendre_polynomials.png')

    # Show the figure
    plt.show()


def experiments_noise_level():
    # Get the current time
    start_time = time.time()

    # Set up the parameters for the experiments
    n_points = np.arange(40, 121, 5)
    noise_levels = np.arange(0.0, 2.1, 0.1)

    print(f'Running {len(n_points) * len(noise_levels)} experiments...')

    if USE_MULTIPROCESSING:
        # Run the experiments in parallel
        settings = [(n_point, noise_level, 10, N_EXPERIMENTS)
                    for noise_level in noise_levels for n_point in n_points]
        with mp.Pool(N_THREADS) as pool:
            values = pool.starmap(run_experiments, settings)
        results = np.array(values).reshape(len(noise_levels), len(n_points))

    else:
        # Create a matrix to store the results
        results = np.zeros((len(noise_levels), len(n_points)))

        # Run the experiments
        experiment_count = 1
        for i, noise_level in enumerate(noise_levels):
            for j, n_point in enumerate(n_points):
                print(f'Running experiment {experiment_count} of {len(n_points) * len(noise_levels)}')
                results[i, j] = run_experiments(n_points=n_point, noise_level=noise_level, target_complexity=10)
                experiment_count += 1

    # Flip the results to match the heatmap
    results = results[::-1, :]

    # Save the results as a csv file
    pd.DataFrame(results, index=noise_levels[::-1], columns=n_points).to_csv('Experiments/noise_level.csv',
                                                                             index_label='Noise Level')

    # Plot the results
    fig, ax = plt.subplots(figsize=(10, 8), tight_layout=True)
    fig.suptitle('Experiments with Noise Level $\\sigma^2$', fontsize=16)
    sns.heatmap(results, ax=ax,
                xticklabels=[str(round(n, 2)) for n in n_points],
                yticklabels=[str(round(n, 2)) for n in noise_levels[::-1]],
                cmap='coolwarm')
    ax.set(xlabel='Number of points', ylabel='Noise level',
           title='Average $E_{out}(g_{10}) - E_{out}(g_{2})$')
    plt.show()

    # Save the figure
    fig.savefig('Figures/lecture_12_noise_level.png')

    # Print the time it took to run the experiments
    print(f'Time taken: {round(time.time() - start_time, 2)}')


def experiments_target_complexity():
    # Get the current time
    start_time = time.time()

    # Set up the parameters for the experiments
    n_points = np.arange(40, 121, 5)
    target_complexities = np.arange(0, 31, 2)

    print(f'Running {len(n_points) * len(target_complexities)} experiments...')

    if USE_MULTIPROCESSING:
        # Run the experiments in parallel
        settings = [(n_point, 0.5, target_complexity, N_EXPERIMENTS)
                    for target_complexity in target_complexities for n_point in n_points]
        with mp.Pool(N_THREADS) as pool:
            values = pool.starmap(run_experiments, settings)
        results = np.array(values).reshape(len(target_complexities), len(n_points))

    else:
        # Create a matrix to store the results
        results = np.zeros((len(target_complexities), len(n_points)))

        # Run the experiments
        experiment_count = 1
        for i, target_complexity in enumerate(target_complexities):
            for j, n_point in enumerate(n_points):
                print(f'Running experiment {experiment_count} of {len(n_points) * len(target_complexities)}')
                results[i, j] = run_experiments(n_points=n_point, noise_level=0.0, target_complexity=target_complexity)
                experiment_count += 1

    # Flip the results to match the heatmap
    results = results[::-1, :]

    # Save the results as a csv file
    pd.DataFrame(results, index=target_complexities[::-1], columns=n_points).to_csv('Experiments/target_complexity.csv',
                                                                                    index_label='Target Complexity')

    # Plot the results
    fig, ax = plt.subplots(figsize=(10, 8), tight_layout=True)
    fig.suptitle('Experiments with Target Complexity $Q_{f}$', fontsize=16)
    sns.heatmap(results, ax=ax,
                xticklabels=[str(round(n, 2)) for n in n_points],
                yticklabels=[str(round(n, 2)) for n in target_complexities[::-1]],
                cmap='coolwarm')
    ax.set(xlabel='Number of points', ylabel='Target complexity',
           title='Average $E_{out}(g_{10}) - E_{out}(g_{2})$')
    plt.show()

    # Save the figure
    fig.savefig('Figures/lecture_12_target_complexity.png')

    # Print the time it took to run the experiments
    print(f'Time taken: {round(time.time() - start_time, 2)}')


if __name__ == '__main__':
    plot_legendre_polynomials()
    experiments_noise_level()
    experiments_target_complexity()
