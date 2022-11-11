import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_dataset_to_dataframe(filename: str) -> pd.DataFrame:
    """
    Load the dataset from the provided file into a DataFrame.

    :param filename: Name of the file to load.
    :return: DataFrame containing the dataset.
    """
    data = pd.read_csv(filename)
    return data


def split_dataset(data: pd.DataFrame, features: List[str],target_column: str) \
        -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split the dataset into training and testing sets. 30% of the data is used for testing.

    :param data: DataFrame containing the dataset.
    :param features: List of features to use.
    :param target_column: Name of the target column.
    :return: Tuple containing the training and testing features and targets.
    """
    X = data[features]
    y = data[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=99)
    return X_train, X_test, y_train, y_test


def calculate_loss(predictions: np.ndarray, targets: np.ndarray) -> float:
    """
    Calculate the mean squared error between the predictions and the targets.

    :param predictions: Predictions made by the model.
    :param targets: Actual targets.
    :return: Mean squared error.
    """
    return metrics.mean_squared_error(targets, predictions)


def widrow_hoff_algorithm(data_points: np.ndarray, targets: np.ndarray, lr: float, max_iterations: int = 1000,
                          mode: str = 'batch') -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Apply the Widrow-Hoff algorithm to the provided data points and targets.

    :param data_points: Data points to use as a NumPy array.
    :param targets: Targets to use as a NumPy array.
    :param lr: Learning rate to use during training.
    :param max_iterations: Maximum number of iterations to run. Defaults to 1000.
    :param mode: Mode to use for training. Either 'batch' or 'sgd'. Defaults to 'batch'.
    :return: Tuple containing the weights and the loss for each iteration.
    """
    # Initialize weights to zeros
    weights = np.zeros(data_points.shape[1])
    num_iterations = 0
    losses = []
    weights_list = [weights]

    while num_iterations < max_iterations:
        # Select the input data according to the mode
        if mode == 'batch':
            input_data = data_points
            target = targets
        elif mode == 'sgd':
            index = np.random.randint(0, data_points.shape[0])
            input_data = data_points[[index]]
            target = targets[[index]]
        else:
            raise ValueError('Mode must be either batch or sgd')

        # Calculate the gradient
        gradient = np.matmul(input_data.T,
                             np.matmul(input_data, weights) - target)
        # Normalize the gradient
        gradient = gradient / np.linalg.norm(gradient)
        # Update the weights
        weights = weights - lr * gradient

        # Calculate the overall loss
        losses.append(calculate_loss(data_points @ weights, targets))
        weights_list.append(weights)

        num_iterations += 1

    return weights, np.array(losses), np.array(weights_list)


def visualize_loss_for_weights(input_data: np.ndarray, targets: np.ndarray, step_size: float = 0.1, levels: int = 20,
                               min_x_weight: float = -15.0, max_x_weight: float = 15.0, xlabel: str = 'x',
                               min_y_weight: float = -15.0, max_y_weight: float = 15.0, ylabel: str = 'y',
                               plot_title: str = 'Loss for different weights',
                               weight_paths: Dict[str, np.ndarray] = None,
                               filename: str = None) -> None:
    """
    Visualize the loss for different weights as contour plot and as 3D surface plot.

    :param weight_paths: Path of the weights during training.
    :param input_data: Input data to use as a NumPy array.
    :param targets: Targets to use as a NumPy array.
    :param step_size: Step size to use for the weights. Defaults to 0.1.
    :param levels: Number of levels to use for the contour plot. Defaults to 20.
    :param min_x_weight: Minimum value for the x-axis weight. Defaults to -15.0.
    :param max_x_weight: Maximum value for the x-axis weight. Defaults to 15.0.
    :param xlabel: Label for the x-axis. Defaults to 'x'.
    :param min_y_weight: Minimum value for the y-axis weight. Defaults to -15.0.
    :param max_y_weight: Maximum value for the y-axis weight. Defaults to 15.0.
    :param ylabel: Label for the y-axis. Defaults to 'y'.
    :param plot_title: Title for the plot. Defaults to 'Loss for different weights'.
    :param filename: Name of the file to save the plot to. If None, the plot is not saved. Defaults to None.
    :return: None
    """
    x_weight_values = np.arange(min_x_weight, max_x_weight, step_size)
    y_weight_values = np.arange(min_y_weight, max_y_weight, step_size)
    X, Y = np.meshgrid(x_weight_values, y_weight_values)

    losses = np.zeros(shape=(x_weight_values.shape[0], y_weight_values.shape[0]))
    for i, x_weight in enumerate(x_weight_values):
        for j, y_weight in enumerate(y_weight_values):
            weights = np.array([x_weight, y_weight])
            predictions = np.matmul(input_data, weights)
            loss = metrics.mean_absolute_error(targets, predictions)
            losses[j, i] = loss

    # Create figure and add subplots
    fig = plt.figure(figsize=(16, 8))
    fig.suptitle(plot_title, fontsize=16)
    ax = fig.add_subplot(121)
    ax2 = fig.add_subplot(122, projection='3d')

    # Plot the loss as a contour plot
    ax.set(xlabel=xlabel, ylabel=ylabel,
           xlim=(min_x_weight, max_x_weight), ylim=(min_y_weight, max_y_weight))
    CS = ax.contour(X, Y, losses, levels=levels)
    ax.clabel(CS, inline=True, fontsize=10)
    if weight_paths is not None:
        for name, weights in weight_paths.items():
            ax.plot(weights[:, 0], weights[:, 1], label=name)
        ax.legend()
    # Plot the loss as a 3D surface
    ax2.set(xlabel='Weight 1', ylabel='Weight 2', zlabel='Loss')
    ax2.plot_surface(X, Y, losses, cmap='viridis', edgecolor='none')

    if filename is not None:
        fig.savefig(f'Figures/assignment_02_{filename}.png')
    plt.show()


def main():
    # Load the dataset, create a scaled version and add a column of ones to both of them
    data = load_dataset_to_dataframe('data/assignment02_wine.csv')
    data_scaled = pd.DataFrame(StandardScaler().fit_transform(data), columns=data.columns, index=data.index)
    data['intersect'] = 1
    data_scaled['intersect'] = 1

    # Set the input data and the targets
    target_column = 'alcohol'
    features = ['residualSugar', 'density']

    # Train the model on the unscaled data
    X_train, X_test, y_train, y_test = split_dataset(data, features, target_column)
    weights, losses, w = widrow_hoff_algorithm(X_train.values, y_train.values, 0.1, 2000, mode='batch')
    print(f'Weights using batch mode: {weights}')
    weights, losses, w2 = widrow_hoff_algorithm(X_train.values, y_train.values, 0.1, 2000, mode='sgd')
    print(f'Weights using sgd mode: {weights}')

    # Visualize the loss without normalization
    visualize_loss_for_weights(data[features].values, data[target_column].values, 0.1,
                               min_x_weight=-15.0, max_x_weight=15.0, xlabel='residualSugar',
                               min_y_weight=-15.0, max_y_weight=15.0, ylabel='density',
                               plot_title='Loss without Normalization', weight_paths={'Batch': w, 'SGD': w2},
                               filename='Unscaled')

    # Train the model on the scaled data
    X_train, X_test, y_train, y_test = split_dataset(data_scaled, features, target_column)
    weights, losses, w = widrow_hoff_algorithm(X_train.values, y_train.values, 0.1, 2000, mode='batch')
    print(f'Weights using batch mode: {weights}')
    weights, losses, w2 = widrow_hoff_algorithm(X_train.values, y_train.values, 0.1, 2000, mode='sgd')
    print(f'Weights using sgd mode: {weights}')

    # Visualize the loss with normalization
    visualize_loss_for_weights(data_scaled[features].values, data_scaled[target_column].values, 0.1,
                               min_x_weight=-5.0, max_x_weight=5.0, xlabel='residualSugar',
                               min_y_weight=-5.0, max_y_weight=5.0, ylabel='density',
                               plot_title='Loss with Normalization', weight_paths={'Batch': w, 'SGD': w2},
                               filename='Scaled')

    # Train the model on all parameters to predict residual sugar
    target_column = 'residualSugar'
    features = ['acidity', 'density', 'pH', 'alcohol', 'intersect']
    X_train, X_test, y_train, y_test = split_dataset(data, features, target_column)

    # Fit the model
    weights, _, _ = widrow_hoff_algorithm(X_train.values, y_train.values, 0.1, 2000, mode='batch')
    print(f'Weights using batch mode: {weights}')

    # Predict both the training and testing set
    y_train_pred = X_train.values @ weights
    print(f'Training set MSE: {metrics.mean_squared_error(y_train, y_train_pred)}')
    y_test_pred = X_test.values @ weights
    print(f'Testing set MSE: {metrics.mean_squared_error(y_test, y_test_pred)}')


if __name__ == '__main__':
    main()
