import numpy as np
from scipy.special import legendre
from .Regression import Regression


def generate_legendre_polynomial(degree: int = 2) -> np.poly1d:
    # Use the generating function to obtain the coefficients
    coefficients = legendre(degree).coef
    # Generate the polynomial
    polynomial = np.poly1d(coefficients)
    return polynomial


def generate_training_set(target_function: np.poly1d, n_points: int, noise: float,
                          lower: float = -1.0, upper: float = 1.0) -> (np.ndarray, np.ndarray):
    # Generate random x values
    x = np.random.uniform(lower, upper, n_points)
    # Generate y values
    y = target_function(x) + np.random.normal(0, noise, n_points)
    return x, y


def calculate_out_of_sample_error(target_function: np.poly1d, x: np.ndarray, y_pred: np.ndarray) -> float:
    # Calculate the out of sample error
    return np.mean((target_function(x) - y_pred) ** 2)


def calculate_area_between_curves(target_function: np.poly1d, x: np.ndarray, y_pred: np.ndarray) -> float:
    # Calculate the area between the curves
    return np.sum(np.abs(target_function(x) - y_pred)) * (x[1] - x[0])


def run_experiments(n_points: int, noise_level: float, target_complexity: int, n_experiments: int = 1000):
    print(f'Running with {n_points} points, noise level {noise_level} and target complexity {target_complexity}')
    # Create a list to store the differences in the out of sample errors
    differences = []

    # Generate the target function
    target_function = generate_legendre_polynomial(degree=target_complexity)

    # Run the experiment n_experiments times
    successful_runs = 0
    while successful_runs < n_experiments:
        # Generate the training set
        x, y = generate_training_set(target_function, n_points, noise_level)

        # Create xticks to evaluate the models
        xticks = np.linspace(min(x), max(x), 100)

        # Fit H2 regression model and calculate the out of sample error
        model = Regression(feature_transform=2).fit(x, y)
        error_h2 = calculate_area_between_curves(target_function, xticks, model.predict(xticks))

        # Fit H10 regression model and calculate the out of sample error
        model = Regression(feature_transform=10).fit(x, y)
        error_h10 = calculate_area_between_curves(target_function, xticks, model.predict(xticks))

        # Calculate the difference in the out of sample errors
        differences.append(error_h10 - error_h2)

        # Increase the number of successful runs
        successful_runs += 1

    # Calculate the average difference in the out of sample errors
    return np.mean(differences)
