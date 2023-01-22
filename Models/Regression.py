from typing import Dict, Union
import numpy as np
from sklearn.metrics import mean_squared_error
from .Model import BaseModel
from sklearn import metrics

LOSS_MAPPING = {'mse': metrics.mean_squared_error, 'mae': metrics.mean_absolute_error}


class RidgeRegression(BaseModel):
    """
    This class implements the ridge regression model.
    """
    def __init__(self, gamma: float = 1.0, delta: float = 0.0, loss: str = 'mse'):
        """
        Initialize the model.

        :param gamma: Gamma value for the RBF kernel.
        :param delta: Lambda value for the ridge regression.
        :param loss: Loss function to use. Either 'mse' or 'mae'.
        """
        # Check the input parameters
        assert isinstance(gamma, (float, int)), 'Gamma must be a numeric.'
        assert isinstance(delta, (float, int)), 'Delta must be a numeric.'
        assert isinstance(loss, str), 'Loss must be a string.'
        assert loss in LOSS_MAPPING.keys(), f'Loss must be one of the following: {LOSS_MAPPING.keys()}'

        # Initialize the parameters
        self.weights: np.ndarray = np.zeros(0)
        self.values: np.ndarray = np.zeros(0)
        self.gamma: float = gamma
        self.delta: float = delta
        self.loss = LOSS_MAPPING[loss]

    def fit(self, features: np.ndarray, labels: np.ndarray) -> 'RidgeRegression':
        """
        Fit the model to the given data. The weights are calculated using the pseudo inverse of the kernel matrix. The
        kernel matrix is calculated using the RBF kernel.

        :param features: Features as a numpy array.
        :param labels: Labels as a numpy array.
        :return: Self.
        """
        # Determine the weights
        self.weights = determine_weights_ridge(x_values=features, y_values=labels, gamma=self.gamma, delta=self.delta)
        self.values = features
        self.is_fitted = True

        # Return the fitted model
        return self

    def predict(self, features: np.ndarray):
        """
        Predict the targets for the given features.

        :param features: Features as a numpy array.
        :return: Predicted targets as a numpy array.
        :raises ValueError: If the model has not been fitted yet.
        """
        if not self.is_fitted:
            raise ValueError("The model has not been fitted yet.")

        # Make prediction for every x value using vectorization
        y_pred = np.array([np.sum(self.weights * np.exp(-self.gamma * (self.values - new_x) ** 2))
                           for new_x in features])
        return y_pred.squeeze()

    def evaluate(self, features: np.ndarray, labels: np.ndarray) -> float:
        """
        Evaluate the model on the given data.

        :param features: Features as a numpy array.
        :param labels: Labels as a numpy array.
        :return: The loss values.
        :raises ValueError: If the model has not been fitted yet.
        """
        if not self.is_fitted:
            raise ValueError("The model has not been fitted yet.")

        # Calculate the loss
        y_pred = self.predict(features).squeeze()
        return self.loss(y_true=labels, y_pred=y_pred)


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


def run_experiment_for_settings(settings: Dict[str, Union[float, int]]) -> float:
    """
    Run multiple experiments for the given settings and return the average MSE.

    :param settings: Dictionary containing the settings.
    :return: Mean of the MSE values.
    """
    # Get the settings
    n_points = settings['n_points']
    n_experiments = settings['n_experiments']
    lower = settings['lower']
    upper = settings['upper']
    gamma = settings['gamma']
    delta = settings['delta']

    # Run the experiment multiple times
    mse = []
    successful_runs = 0
    while successful_runs < n_experiments:
        try:
            # Sample x values and calculate the respective y with noise
            x_values = np.random.uniform(low=lower, high=upper, size=n_points)
            y_values = x_values ** 2 + np.random.normal(loc=0, scale=0.1, size=n_points)

            # Create the model and fit it
            model = RidgeRegression(gamma=gamma, delta=delta).fit(features=x_values, labels=y_values)

            # Make predictions
            x_ticks = np.linspace(lower, upper, 200)
            mse.append(mean_squared_error(y_true=x_ticks ** 2, y_pred=model.predict(x_ticks)))
            successful_runs += 1
        except np.linalg.LinAlgError:
            pass

    return np.mean(mse)
