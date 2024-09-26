from abc import ABC, abstractmethod
import numpy as np
from .constants import Constants


class Perceptron(ABC):
    def __init__(self, activationFunction, weights):
        self.activationFunction = activationFunction
        self.weights = weights
        self.weightsCount = len(weights)
        self.weightsHistory = np.array([weights])
        self.constants = Constants.getInstance()

    @abstractmethod
    def train(self, inputs, expectedOutputs):
        pass

    @abstractmethod
    def calculateError(self, inputs, expectedOutputs):
        pass

    @abstractmethod
    def test(self, input):
        pass


class SingleLayerPerceptron(Perceptron):
    def train(self, inputs, expectedOutputs):
        for _ in range(self.constants.maxEpochs):
            for input, expectedOutput in zip(inputs, expectedOutputs):
                output = self.activationFunction(input, self.weights)
                deltaW = self.constants.learningRate * (expectedOutput - output) * self.activationFunction.derivative(input, self.weights) * input
                self.weights += deltaW
                self.weightsHistory.append(np.copy(self.weights))

                if self.calculateError(inputs, expectedOutputs) <= self.constants.epsilon:
                    return

    def calculateError(self, inputs, expectedOutputs):
        error = 0
        for input, expectedOutput in zip(inputs, expectedOutputs):
            error += np.power(expectedOutput - self.activationFunction(input, self.weights), 2)
        return error / len(expectedOutputs)

    def test(self, input):
        return self.activationFunction(input, self.weights)


class MultiLayerPerceptron(Perceptron):
    pass
