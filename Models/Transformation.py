import numpy as np


def transform_to_polynomial(x: np.ndarray, degree: int) -> np.ndarray:
    """
    Transform the given data to a polynomial of the given degree.

    :param x: The data to transform.
    :param degree: The degree of the polynomial.
    :return: The transformed data.
    """
    return np.array([x ** i for i in range(degree + 1)]).T
