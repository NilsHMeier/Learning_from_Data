import abc
import numpy as np


class BaseModel(abc.ABC):
    @abc.abstractmethod
    def __init__(self):
        self.is_fitted = False

    @abc.abstractmethod
    def fit(self, features: np.ndarray, labels: np.ndarray):
        self.is_fitted = True

    @abc.abstractmethod
    def predict(self, features: np.ndarray):
        pass

    def fit_predict(self, features, labels):
        self.fit(features, labels)
        return self.predict(features)

    @abc.abstractmethod
    def evaluate(self, features: np.ndarray, labels: np.ndarray) -> float:
        pass
