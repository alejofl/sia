from abc import ABC, abstractmethod
import numpy as np
from .constants import Constants


class ActivationFunction(ABC):
    @abstractmethod
    def __call__(self, x, w):
        pass

    @abstractmethod
    def derivative(self, x, w):
        pass


class StepFunction(ActivationFunction):
    def __call__(self, x, w):
        w0 = w[0]
        w = w[1:]
        value = np.sign(np.dot(w, x) - w0)
        if value == 0:
            return 1
        return value

    def derivative(self, x, w):
        return 1


class LinearFunction(ActivationFunction):
    def __call__(self, x, w):
        w0 = w[0]
        w = w[1:]
        return np.dot(w, x) + w0

    def derivative(self, x, w):
        return 1


class LogisticFunction(ActivationFunction):
    def __call__(self, x, w):
        beta = Constants.getInstance().beta    
        w0 = w[0]
        w = w[1:]
        return 1 / (1 + np.exp(-2 * beta * (np.dot(w, x) + w0)))

    def derivative(self, x, w):
        beta = Constants.getInstance().beta    
        return 2 * beta * self.__call__(x, w) * (1 - self.__call__(x, w))


class HyperbolicTangentFunction(ActivationFunction):
    def __call__(self, x, w):
        beta = Constants.getInstance().beta
        w0 = w[0]
        w = w[1:]
        return np.tanh(beta * (np.dot(w, x) + w0))

    def derivative(self, x, w):
        beta = Constants.getInstance().beta
        return beta * (1 - np.power(self.__call__(x, w), 2))
