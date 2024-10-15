from abc import ABC, abstractmethod
import numpy as np


class DistanceCalculator(ABC):
    @abstractmethod
    def __call__(self, x, y):
        pass

    @staticmethod
    def getCalculator(name):
        name = name.lower()
        if name == "euclidean":
            return EuclideanDistance()
        elif name == "exponential":
            return ExponentialDistance()
        else:
            raise ValueError("Invalid distance calculator type")


class EuclideanDistance(DistanceCalculator):
    def __call__(self, x, y):
        return np.linalg.norm(x - y)


class ExponentialDistance(DistanceCalculator):
    def __call__(self, x, y):
        return np.exp(-1 * np.power(np.linalg.norm(x - y), 2))
