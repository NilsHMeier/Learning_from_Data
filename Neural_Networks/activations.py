from typing import List, Union
import numpy as np
from .util import cast_inputs


def linear(x: Union[float, List[float], np.ndarray]) -> np.ndarray:
    """
    Calculates a linear activation function y = x.

    :param x: The input to the activation function.
    :return: The output of the activation function.
    """
    x = cast_inputs(x)
    return x


def linear_derivative(x: Union[float, List[float], np.ndarray]) -> np.ndarray:
    """
    Calculates the derivative of a linear activation function y = x.

    :param x: The input to the activation function.
    :return: The derivative of the activation function.
    """
    x = cast_inputs(x)
    return np.ones_like(x)


def sign(x: Union[float, List[float], np.ndarray]) -> np.ndarray:
    """
    Calculates a sign activation function y = 1 if x >= 0 else -1.

    :param x: The input to the activation function.
    :return: The output of the activation function.
    """
    x = cast_inputs(x)
    return np.where(x >= 0, 1, -1)


def sign_derivative(x: Union[float, List[float], np.ndarray]) -> np.ndarray:
    """
    Calculates the derivative of a sign activation function y = 1 if x >= 0 else -1.

    :param x: The input to the activation function.
    :return: The derivative of the activation function.
    """
    x = cast_inputs(x)
    return np.zeros_like(x)


def relu(x: Union[float, List[float], np.ndarray]) -> np.ndarray:
    """
    Calculates a rectified linear unit activation function y = max(0, x).

    :param x: The input to the activation function.
    :return: The output of the activation function.
    """
    x = cast_inputs(x)
    return np.where(x > 0, x, 0)


def relu_derivative(x: Union[float, List[float], np.ndarray]):
    """
    Calculates the derivative of a rectified linear unit activation function y = max(0, x).

    :param x: The input to the activation function.
    :return: The derivative of the activation function.
    """
    x = cast_inputs(x)
    return np.where(x > 0, 1, 0)


def sigmoid(x: Union[float, List[float], np.ndarray]) -> np.ndarray:
    """
    Calculates a sigmoid activation function y = 1 / (1 + e^-x).

    :param x: The input to the activation function.
    :return: The output of the activation function.
    """
    x = cast_inputs(x)
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x: Union[float, List[float], np.ndarray]) -> np.ndarray:
    """
    Calculates the derivative of a sigmoid activation function y = 1 / (1 + e^-x).

    :param x: The input to the activation function.
    :return: The derivative of the activation function.
    """
    # x = cast_inputs(x)
    return sigmoid(x) * (1 - sigmoid(x))


def tanh(x: Union[float, List[float], np.ndarray]) -> np.ndarray:
    """
    Calculates a hyperbolic tangent activation function y = tanh(x).

    :param x: The input to the activation function.
    :return: The output of the activation function.
    """
    x = cast_inputs(x)
    return np.tanh(x)


def tanh_derivative(x: Union[float, List[float], np.ndarray]) -> np.ndarray:
    """
    Calculates the derivative of a hyperbolic tangent activation function y = tanh(x).

    :param x: The input to the activation function.
    :return: The derivative of the activation function.
    """
    x = cast_inputs(x)
    return 1 - np.tanh(x) ** 2


def get_activation_function(activation: str):
    """
    Returns the respective activation function based on the string name.

    :param activation: The name of the activation function.
    :return: The activation function reference.
    """
    if activation == 'linear':
        return linear
    elif activation == 'sign':
        return sign
    elif activation == 'relu':
        return relu
    elif activation == 'sigmoid':
        return sigmoid
    elif activation == 'tanh':
        return tanh
    else:
        raise ValueError('Activation function not found.')


def get_activation_derivative(activation: str):
    """
    Returns the respective activation function derivative based on the string name.

    :param activation: The name of the activation function.
    :return: The activation function derivative reference.
    """
    if activation == 'linear':
        return linear_derivative
    elif activation == 'sign':
        return sign_derivative
    elif activation == 'relu':
        return relu_derivative
    elif activation == 'sigmoid':
        return sigmoid_derivative
    elif activation == 'tanh':
        return tanh_derivative
    else:
        raise ValueError('Activation function not found.')
