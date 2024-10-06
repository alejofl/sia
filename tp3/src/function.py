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
        if name == "STEP":
            return StepFunction(options)
        elif name == "LINEAR":
            return LinearFunction(options)
        elif name == "LOGISTIC":
            return LogisticFunction(options)
        elif name == "TANH":
            return HyperbolicTangentFunction(options)
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
    
    def normalize(self, x):
        return x
    
    def denormalize(self, x):
        return x


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
        return ((x-self.min)/(self.max-self.min))*(1-0)+0
    
    def denormalize(self, x):
        return (((x-0)*(self.max-self.min))/(1-0))+self.min


class HyperbolicTangentFunction(ActivationFunction):
    def __call__(self, x, w):
        beta = self.options["beta"]
        return np.tanh(beta * (np.dot(w, x)))

    def derivative(self, x, w):
        beta = self.options["beta"]
        return beta * (1 - np.power(self.__call__(x, w), 2))
    
    def normalize(self, x):
        return ((x-self.min)/(self.max-self.min))*(1-(-1))-1
    
    def denormalize(self, x):
        return (((x-(-1))*(self.max-self.min))/(1-(-1)))+self.min
