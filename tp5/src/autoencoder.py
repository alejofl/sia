import numpy as np
from .perceptron import MultiLayerPerceptron
from .utils import Utils


class Autoencoder(MultiLayerPerceptron):
    def __init__(self, encoderArchitecture, optimizerClass, optimizerOptions, inputs):
        self.latentSpaceIndex = len(encoderArchitecture) - 1
        architecture = Utils.generateAutoencoderArchitecture(encoderArchitecture, len(inputs[0]))
        self.inputs = inputs
        super().__init__(architecture, optimizerClass, optimizerOptions)

    def train(self):
        expectedOutputs = []
        for input in self.inputs:
            expectedOutputs.append(input[1:])
        super().train(self.inputs, expectedOutputs)

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