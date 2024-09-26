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
    
    @staticmethod
    def getFunction(name):
        if name == "STEP":
            return StepFunction()
        elif name == "LINEAR":
            return LinearFunction()
        elif name == "LOGISTIC":
            return LogisticFunction()
        elif name == "TANH":
            return HyperbolicTangentFunction()
        else:
            raise ValueError("Invalid activation function")


class StepFunction(ActivationFunction):
    def __call__(self, x, w):
        value = np.sign(np.dot(w, x))
        if value == 0:
            return 1
        return value

    def derivative(self, x, w):
        return 1


class LinearFunction(ActivationFunction):
    def __call__(self, x, w):
        return np.dot(w, x)

    def derivative(self, x, w):
        return 1


class LogisticFunction(ActivationFunction):
    def __call__(self, x, w):
        beta = Constants.getInstance().beta
        return 1 / (1 + np.exp(-2 * beta * (np.dot(w, x))))

    def derivative(self, x, w):
        beta = Constants.getInstance().beta
        return 2 * beta * self.__call__(x, w) * (1 - self.__call__(x, w))


class HyperbolicTangentFunction(ActivationFunction):
    def __call__(self, x, w):
        beta = Constants.getInstance().beta
        return np.tanh(beta * (np.dot(w, x)))

    def derivative(self, x, w):
        beta = Constants.getInstance().beta
        return beta * (1 - np.power(self.__call__(x, w), 2))
