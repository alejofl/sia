from abc import ABC, abstractmethod


class Perceptron(ABC):
    def __init__(self, activationFunction, learningRate, maxEpochs, weights):
        self.activationFunction = activationFunction
        self.learningRate = learningRate
        self.maxEpochs = maxEpochs
        self.weights = weights
        self.weightCount = len(weights)
        self.weightsHistory = [weights]
        
    @abstractmethod
    def train(self, inputs, expectedOutputs):
        pass

    @abstractmethod
    def test(self, input):
        pass


class SingleLayerPerceptron(Perceptron):
    pass


class MultiLayerPerceptron(Perceptron):
    pass
