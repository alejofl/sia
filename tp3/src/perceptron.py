from abc import ABC, abstractmethod
import numpy as np
from .constants import Constants
from .function import ActivationFunction


class Perceptron(ABC):
    def __init__(self, activationFunction: ActivationFunction, weights):
        self.activationFunction = activationFunction
        self.weights = weights
        self.weightsCount = len(weights)
        self.weightsHistory = [weights]
        self.weightsPerEpoch = []
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
        self.activationFunction.configureNormalization(expectedOutputs)
        for _ in range(self.constants.maxEpochs):
            for input, expectedOutput in zip(inputs, expectedOutputs):
                output = self.activationFunction(input, self.weights)
                deltaW = self.constants.learningRate * (self.activationFunction.normalize(expectedOutput) - output) * self.activationFunction.derivative(input, self.weights) * input
                self.weights = np.add(self.weights, deltaW)
                self.weightsHistory.append(np.copy(self.weights))

                if np.abs(self.calculateError(inputs, expectedOutputs)) <= self.constants.epsilon:
                    return
            self.weightsPerEpoch.append(np.copy(self.weights))
        print("Training finished without convergence.")

    def calculateError(self, inputs, expectedOutputs):
        error = 0
        for input, expectedOutput in zip(inputs, expectedOutputs):
            error += np.power(self.activationFunction.normalize(expectedOutput) - self.activationFunction(input, self.weights), 2)
        return error / len(expectedOutputs)

    def test(self, input):
        return self.activationFunction.denormalize(self.activationFunction(input, self.weights))


class MultiLayerPerceptron(Perceptron):
    pass
