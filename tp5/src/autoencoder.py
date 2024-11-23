import numpy as np
from .perceptron import MultiLayerPerceptron
from .utils import Utils


class Autoencoder(MultiLayerPerceptron):
    def __init__(self, encoderArchitecture, optimizerClass, optimizerOptions, inputs, isVae=False):
        self.latentSpaceIndex = len(encoderArchitecture) - 1
        self.latentSpaceDimension = encoderArchitecture[-1]["neuronQty"]
        architecture = Utils.generateAutoencoderArchitecture(encoderArchitecture, len(inputs[0]), False, isVae)
        self.inputs = inputs
        self.expectedOutputs = []
        for input in self.inputs:
            self.expectedOutputs.append(input[1:])
        super().__init__(architecture, optimizerClass, optimizerOptions)

    def train(self):
        super().train(self.inputs, self.expectedOutputs)

    def getLatentSpaceOutput(self, input):
        outputs = []
        for i in range(self.latentSpaceIndex + 1):
            layer = self.layers[i]
            layerInput = outputs[i-1] if i>0 else input
            layerOutput = np.array(list(map(lambda n: n.test(layerInput), layer)))
            if i != len(self.layers)-1:
                layerOutput = np.insert(layerOutput, 0, self.constants.bias)
            outputs.append(layerOutput)
        return outputs[-1]

    def testWithLatentSpaceInput(self, input):
        if len(input) != self.latentSpaceDimension:
            raise ValueError("Input must have the same dimension as the latent space")

        input = np.insert(input, 0, self.constants.bias)
        outputs = []
        for idx, i in enumerate(range(self.latentSpaceIndex + 1, len(self.layers))):
            layer = self.layers[i]
            layerInput = outputs[idx - 1] if idx>0 else input
            layerOutput = np.array(list(map(lambda n: n.test(layerInput), layer)))
            if i != len(self.layers)-1:
                layerOutput = np.insert(layerOutput, 0, self.constants.bias)
            outputs.append(layerOutput)
        return outputs[-1]
