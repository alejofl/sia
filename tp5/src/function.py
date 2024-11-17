from abc import ABC, abstractmethod
import numpy as np


class ActivationFunction(ABC):
    def __init__(self, options=None):
        self.options = options

    @abstractmethod
    def __call__(self, x, w):
        pass

    @abstractmethod
    def derivative(self, x, w):
        pass

    @abstractmethod
    def normalize(self, x):
        pass

    @abstractmethod
    def denormalize(self, x):
        pass

    def configureNormalization(self, min, max):
        self.min = min
        self.max = max
    
    @staticmethod
    def getFunction(name, options=None):
        name = name.lower()
        if name == "linear":
            return LinearFunction(options)
        elif name == "logistic":
            return LogisticFunction(options)
        elif name == "tanh":
            return HyperbolicTangentFunction(options)
        elif name == "relu":
            return ReluFunction(options)
        else:
            raise ValueError("Invalid activation function")


class LinearFunction(ActivationFunction):
    def __call__(self, x, w):
        return np.dot(w, x)

    def derivative(self, x, w):
        return 1
    
    def normalize(self, x):
        return x
    
    def denormalize(self, x):
        return x


class LogisticFunction(ActivationFunction):
    def __call__(self, x, w):
        beta = self.options["beta"]
        return 1 / (1 + np.exp(-2 * beta * (np.dot(w, x))))

    def derivative(self, x, w):
        beta = self.options["beta"]
        return 2 * beta * self.__call__(x, w) * (1 - self.__call__(x, w))
    
    def normalize(self, x):
        return np.interp(x, [self.min, self.max], [0, 1])
    
    def denormalize(self, x):
        return np.interp(x, [0, 1], [self.min, self.max])


class HyperbolicTangentFunction(ActivationFunction):
    def __call__(self, x, w):
        beta = self.options["beta"]
        return np.tanh(beta * (np.dot(w, x)))

    def derivative(self, x, w):
        beta = self.options["beta"]
        return beta * (1 - np.power(self.__call__(x, w), 2))
    
    def normalize(self, x):
        return np.interp(x, [self.min, self.max], [-1, 1])
    
    def denormalize(self, x):
        return np.interp(x, [-1, 1], [self.min, self.max])

class ReluFunction(ActivationFunction):
    def __call__(self, x, w):
        return np.max([0, np.dot(w, x)])

    def derivative(self, x, w):
        return 1 if np.dot(w, x) > 0 else 0
    
    def normalize(self, x):
        return x
    
    def denormalize(self, x):
        return x
