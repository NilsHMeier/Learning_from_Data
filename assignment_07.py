import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Set the number of samples to generate
N_RUNS = 100


def simulate_h0(data_points: np.ndarray):
    # Create ticks for plotting and use them as DataFrame Index
    x_ticks = np.linspace(-1, 1, 200)
    result_df = pd.DataFrame(index=x_ticks)
    run = 1
    # Run over all datasets (containing two x values each)
    for x1, x2 in data_points:
        # Calculate the target values
        y1, y2 = np.sin(np.pi * x1), np.sin(np.pi * x2)
        # Take the mean and add it to the DataFrame
        y_mean = (y1 + y2) / 2
        result_df[f'Run_{run}'] = y_mean
        run += 1

    # Plot the results
    plot_results(result_df, title='Mean Hypothesis')


def simulate_h1(data_points: np.ndarray):
    # Create ticks for plotting and use them as DataFrame Index
    x_ticks = np.linspace(-1, 1, 200)
    result_df = pd.DataFrame(index=x_ticks)
    run = 1
    # Run over all datasets (containing two x values each)
    for x1, x2 in data_points:
        # Calculate the target values
        y1, y2 = np.sin(np.pi * x1), np.sin(np.pi * x2)
        # Calculate line params and put respective values to the DataFrame
        slope = (y1 - y2) / (x1 - x2)
        intercept = y1 - slope * x1
        result_df[f'Run_{run}'] = x_ticks * slope + intercept
        run += 1

    # Plot the results
    plot_results(result_df, title='Line Hypothesis')


def plot_results(result_df: pd.DataFrame, title: str):
    # Extract x values and calculate mean & std
    x_ticks = result_df.index.values
    mean_values = result_df.mean(axis=1)
    std_values = result_df.std(axis=1)

    # Calculate bias and variance
    bias = np.abs(mean_values - np.sin(x_ticks)).mean()
    var = result_df.var(axis=1).mean()

    # Plot the results
    fig, axs = plt.subplots(ncols=2, figsize=(12, 6))
    fig.suptitle(title, size=16)

    # Plot the lines in the first plot
    for col in result_df.columns:
        axs[0].plot(x_ticks, result_df[col], color='gray')
    axs[0].plot(x_ticks, np.sin(np.pi * x_ticks), color='blue')
    axs[0].set(title=f'Bias = {round(bias, 2)}')

    # Plot the mean values and area in second plot
    axs[1].fill_between(x_ticks, mean_values + std_values, mean_values - std_values, color='gray')
    axs[1].plot(x_ticks, np.sin(np.pi * x_ticks), color='blue')
    axs[1].plot(x_ticks, mean_values, color='red')
    axs[1].set(title=f'Variance = {round(var, 2)}')

    plt.tight_layout()
    plt.show()
    fig.savefig(f'Figures/assignment_07_{title}.png')


def main():
    # Create a random sample of x values
    np.random.seed(42)
    sample = np.random.uniform(low=-1, high=1, size=[N_RUNS, 2])
    simulate_h0(data_points=sample)
    simulate_h1(data_points=sample)


if __name__ == '__main__':
    main()
