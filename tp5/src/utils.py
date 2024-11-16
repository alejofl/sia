import numpy as np
from .constants import Constants
from .function import ActivationFunction


class Utils:
    @staticmethod
    def parseFont(font):
        letters = []
        for f in font:
            bits = []
            for row in f:
                bits.append((row & 0b00010000) >> 4)
                bits.append((row & 0b00001000) >> 3)
                bits.append((row & 0b00000100) >> 2)
                bits.append((row & 0b00000010) >> 1)
                bits.append(row & 0b00000001)
            letters.append(np.array(bits))
        return letters

    @staticmethod
    def generateInputs(letters):
        bias = Constants.getInstance().bias
        inputs = []
        for letter in letters:
            input = np.array([letter])
            inputs.append(np.insert(input, 0, bias))
        return inputs

    @staticmethod
    def generateAutoencoderArchitecture(encoderArchitecture, inputLength, zeros=False):
        r = Constants.getInstance().random
        outputLayer = { "neuronQty": inputLength - 1, "activationFunction": { "type": "LOGISTIC", "options": { "beta": 0.5 } } }
        layers = encoderArchitecture + list(reversed(encoderArchitecture[:-1])) + [outputLayer]

        architecture = []
        i = 0
        for i, layer in enumerate(layers):
            neuronQty = layer["neuronQty"]
            f = ActivationFunction.getFunction(layer["activationFunction"]["type"], layer["activationFunction"]["options"])
            weights = []
            for _ in range(neuronQty):
                weightsQty = layers[i - 1]["neuronQty"] + 1
                weights.append(np.zeros(weightsQty) if zeros else r.uniform(-1, 1, weightsQty))
            architecture.append((neuronQty, f, weights))
        return architecture

    @staticmethod
    def postprocessOutput(output):
        return list(map(lambda x: 1 if x > 0.5 else 0, output))
