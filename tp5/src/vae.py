import numpy as np
from .utils import Utils
from .autoencoder import Autoencoder


class VAE(Autoencoder):
    def __init__(self, encoderArchitecture, optimizerClass, optimizerOptions, inputs):
        super().__init__(encoderArchitecture, optimizerClass, optimizerOptions, inputs, True)

    def train(self):
        seenInputs = 0
        for epoch in range(self.constants.maxEpochs):
            totalLoss = 0

            for input, expectedOutput in zip(self.inputs, self.expectedOutputs):
                seenInputs += 1

                # Forward pass
                outputs = []
                for i, layer in enumerate(self.layers):
                    layerInput = outputs[i-1] if i > 0 else input
                    layerOutput = np.array(list(map(lambda n: n.test(layerInput), layer)))
                    if i == self.latentSpaceIndex: # I'm on the latent space
                        mu = layerOutput[:self.latentSpaceDimension]
                        sigma = layerOutput[self.latentSpaceDimension:]
                        # Reparametrization trick
                        epsilon, layerOutput = Utils.sample(mu, sigma)
                    if i != len(self.layers) - 1:
                        layerOutput = np.insert(layerOutput, 0, self.constants.bias)
                    outputs.append(layerOutput)
                
                reconstructionLoss = 0.5 * np.mean((input[1:] - outputs[-1]) ** 2)
                kl = -0.5 * np.sum(1 + sigma - mu**2 - np.exp(sigma))
                totalLoss += reconstructionLoss + kl
                
                # Backpropagation
                deltas = []
                for i, layer in reversed(list(enumerate(self.layers))):
                    layerDeltas = []
                    for k, n in enumerate(layer):
                        delta = 0
                        layerInput = outputs[i-1] if i>0 else input
                        if i == len(self.layers) - 1:
                            delta = (expectedOutput[k] - outputs[i][k]) * n.activationFunction.derivative(layerInput, n.weights)
                        elif i == self.latentSpaceIndex:
                            encoderRecError = np.concatenate([np.dot(deltas[-1], 1), np.dot(deltas[-1], epsilon)])
                            encoderKLError = np.concatenate([mu, (1 - np.exp(sigma))*(-0.5)])
                            delta = (encoderRecError[k] + encoderKLError[k]) * n.activationFunction.derivative(layerInput, n.weights)
                        else:
                            weightsBetweenMeAndNextLayer = np.array([r.weights[k+1] for r in self.layers[i+1]])
                            delta = np.dot(deltas[-1], weightsBetweenMeAndNextLayer) * n.activationFunction.derivative(layerInput, n.weights)
                        layerDeltas.append(delta)
                        deltaW = n.optimizer(delta * layerInput, previousDeltaW=n.weights - n.weightsHistory[-2] if len(n.weightsHistory) > 1 else 0)
                        n.incrementDeltaW(deltaW)
                        
                    deltas.append(layerDeltas)

                if self.constants.batchSize != -1 and seenInputs % self.constants.batchSize == 0:
                    for layer in self.layers:
                        for n in layer:
                            n.updateWeights()
                    if np.abs(totalLoss) <= self.constants.epsilon:
                        for layer in self.layers:
                            for neuron in layer:
                                neuron.saveWeightsForEpoch()
                        return

            if self.constants.batchSize == -1:
                for layer in self.layers:
                    for n in layer:
                        n.updateWeights()
                if np.abs(totalLoss) <= self.constants.epsilon:
                    for layer in self.layers:
                        for neuron in layer:
                            neuron.saveWeightsForEpoch()
                    return

            for layer in self.layers:
                for neuron in layer:
                    neuron.saveWeightsForEpoch()
            
            # TODO: Remove this print
            if epoch % 200 == 0:
                print("epoch", epoch, totalLoss)
        print("Training finished without convergence.")

    def calculateError(self, inputs, expectedOutputs):
        raise ValueError("Not usable in VAE")

    def test(self, input):
        outputs = []
        for i, layer in enumerate(self.layers):
            layerInput = outputs[i-1] if i>0 else input
            layerOutput = np.array(list(map(lambda n: n.test(layerInput), layer)))
            if i == self.latentSpaceIndex:
                mu = layerOutput[:self.latentSpaceDimension]
                sigma = layerOutput[self.latentSpaceDimension:]
                _, layerOutput = Utils.sample(mu, sigma)
            if i != len(self.layers)-1:
                layerOutput = np.insert(layerOutput, 0, self.constants.bias)
            outputs.append(layerOutput)
        return outputs[-1]

    def getLatentSpaceOutput(self, input):
        outputs = []
        for i in range(self.latentSpaceIndex + 1):
            layer = self.layers[i]
            layerInput = outputs[i-1] if i>0 else input
            layerOutput = np.array(list(map(lambda n: n.test(layerInput), layer)))
            if i == self.latentSpaceIndex:
                mu = layerOutput[:self.latentSpaceDimension]
                sigma = layerOutput[self.latentSpaceDimension:]
                _, layerOutput = Utils.sample(mu, sigma)
            if i != len(self.layers)-1:
                layerOutput = np.insert(layerOutput, 0, self.constants.bias)
            outputs.append(layerOutput)
        return outputs[-1]
