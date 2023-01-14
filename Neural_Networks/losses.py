from typing import List, Union
import numpy as np
from .util import cast_inputs


def mean_squared_error(predictions: Union[float, List[float], np.ndarray],
                       labels: Union[float, List[float], np.ndarray]) -> float:
    """
    Calculates the mean squared error between predictions and labels.

    :param predictions: The predictions of the model.
    :param labels: The labels of the data.
    :return: The mean squared error between predictions and labels.
    """
    predictions = cast_inputs(predictions)
    labels = cast_inputs(labels)
    assert len(predictions) == len(labels), 'Predictions and labels must be the same length.'
    return np.sum((predictions.squeeze() - labels) ** 2) / len(predictions)


def mean_squared_error_derivative(prediction, label, signal, activation_derivative):
    return 2 * (prediction - label) * activation_derivative(signal)


def get_loss_function(loss_function_name: str):
    if loss_function_name in ['mean_squared_error', 'mse', 'MSE']:
        return mean_squared_error, mean_squared_error_derivative
    else:
        raise ValueError('Loss function not found.')
