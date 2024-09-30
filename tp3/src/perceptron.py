from abc import ABC, abstractmethod
import numpy as np
from .constants import Constants
from .function import ActivationFunction


class Perceptron(ABC):
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
    def __init__(self, activationFunction: ActivationFunction, weights):
        self.activationFunction = activationFunction
        self.weights = weights
        self.deltaW = np.zeros(len(weights))
        self.weightsCount = len(weights)
        self.weightsHistory = [weights]
        self.weightsPerEpoch = []
        self.constants = Constants.getInstance()

    def train(self, inputs, expectedOutputs):
        self.activationFunction.configureNormalization(expectedOutputs)
        for _ in range(self.constants.maxEpochs):
            seenInputs = 0
            for input, expectedOutput in zip(inputs, expectedOutputs):
                seenInputs += 1

                output = self.activationFunction(input, self.weights)
                deltaW = self.constants.learningRate * (self.activationFunction.normalize(expectedOutput) - output) * self.activationFunction.derivative(input, self.weights) * input
                self.incrementDeltaW(deltaW)

                if self.constants.updateWeightsEveryXInputs != -1 and seenInputs % self.constants.updateWeightsEveryXInputs == 0:
                    self.updateWeights()
                    if np.abs(self.calculateError(inputs, expectedOutputs)) <= self.constants.epsilon:
                        return

            if self.constants.updateWeightsEveryXInputs == -1:
                self.updateWeights()
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

    def incrementDeltaW(self, deltaW):
        self.deltaW = np.add(self.deltaW, deltaW)

    def updateWeights(self):
        self.weights = np.add(self.weights, self.deltaW)
        self.weightsHistory.append(np.copy(self.weights))
        self.deltaW = np.zeros(self.weightsCount)


 # architecture: [ [neuronQty, actFn, [[w11, w12, ..], [w21, w22, ...]]], .... ]
class MultiLayerPerceptron(Perceptron):
    def __init__(self, architecture):
        self.layers = []
        for qty, activationFn, weights in architecture:
            layer = []
            for i in range(qty):
                n = SingleLayerPerceptron(activationFn, weights[i])
                layer.append(n)
            self.layers.append(layer)
        self.constants = Constants.getInstance()

    def train(self, inputs, expectedOutputs):
        # TODO: define normalization
        for _ in range(self.constants.maxEpochs):
            seenInputs = 0
            for input, expectedOutput in zip(inputs, expectedOutputs):
                seenInputs += 1

                outputs = np.array()
                for i, layer in enumerate(self.layers):
                    layerInput = outputs[i-1] if i>0 else input
                    layerOutput = map(lambda n: n.test(layerInput), layer)
                    outputs = np.append(outputs, [layerOutput])

                deltas = []
                for i, layer in reversed(list(enumerate(self.layers))):
                    layerDeltas = []
                    for k, n in enumerate(layer):
                        delta = 0
                        if i == len(self.layers):
                            delta = (expectedOutput[k] - outputs[i][k]) * n.activationFunction.derivative(outputs[i-1], n.weights)
                        else:
                            delta = np.dot(deltas[-1], self.layers[i+1][k].weights) * n.activationFunction.derivative(outputs[i-1], n.weights)
                        layerDeltas.append(delta)
                        deltaW = self.constants.learningRate * delta * outputs[i-1]
                        n.incrementDeltaW(deltaW)
                    deltas.append(layerDeltas)

                if self.constants.updateWeightsEveryXInputs != -1 and seenInputs % self.constants.updateWeightsEveryXInputs == 0:
                    for layer in self.layers:
                        for n in layer:
                            n.updateWeights()
                    if np.abs(self.calculateError(inputs, expectedOutputs)) <= self.constants.epsilon:
                        return

            if self.constants.updateWeightsEveryXInputs == -1:
                for layer in self.layers:
                    for n in layer:
                        n.updateWeights()
                if np.abs(self.calculateError(inputs, expectedOutputs)) <= self.constants.epsilon:
                    return
        print("Training finished without convergence.")

    def calculateError(self, inputs, expectedOutputs):
        error = 0
        qty = 0
        for input, expectedOutput in zip(inputs, expectedOutputs):
            output = self.test(input)
            for o, eo in zip(output, expectedOutput):
                qty += 1
                error += np.power(eo - o, 2)
        return error / qty

    def test(self, input):
        outputs = np.array()
        for i, layer in enumerate(self.layers):
            layerInput = outputs[i-1] if i>0 else input
            layerOutput = map(lambda n: n.test(layerInput), layer)
            outputs = np.append(outputs, [layerOutput])
        return outputs[-1]
