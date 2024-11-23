import numpy as np
from .constants import Constants
from .function import ActivationFunction


class Utils:
    @staticmethod
    def parseBoolean(value):
        return value.lower() in ['true', '1', 't', 'y', 'yes']

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
    def generateInput(letter):
        bias = Constants.getInstance().bias
        input = np.array([letter])
        return np.insert(input, 0, bias)

    @staticmethod
    def generateInputs(letters):
        inputs = []
        for letter in letters:
            inputs.append(Utils.generateInput(letter))
        return inputs

    @staticmethod
    def generateAutoencoderArchitecture(encoderArchitecture, inputLength, zeros=False, isVae=False):
        r = Constants.getInstance().random
        outputLayer = { "neuronQty": inputLength - 1, "activationFunction": { "type": "LOGISTIC", "options": { "beta": 1 } } }
        layers = encoderArchitecture + list(reversed(encoderArchitecture[:-1])) + [outputLayer]
        print(isVae)
        if isVae:
            print(layers[len(encoderArchitecture)-1]["neuronQty"])
            layers[len(encoderArchitecture)-1]["neuronQty"]*=2
        architecture = []
        i = 0
        for i, layer in enumerate(layers):
            neuronQty = layer["neuronQty"]
            f = ActivationFunction.getFunction(layer["activationFunction"]["type"], layer["activationFunction"]["options"])
            weights = []
            for _ in range(neuronQty):
                weightsQty = layers[i - 1]["neuronQty"] + 1
                if isVae and i == len(encoderArchitecture):
                    weightsQty = weightsQty//2 + 1

                weights.append(np.zeros(weightsQty) if zeros else r.uniform(-1, 1, weightsQty))
            architecture.append((neuronQty, f, weights))
        print("architecture:", len(architecture[0][2]))
        return architecture

    @staticmethod
    def postprocessOutput(output, binarize=False):
        if binarize:
            return list(map(lambda x: 1 if x > 0.5 else 0, output))
        return output

    @staticmethod
    def saltAndPepperFilter(data, noiseThreshold):
        random = Constants.getInstance().random
        noisyData = np.copy(data)
        for i in range(len(noisyData)):
            if random.random() < noiseThreshold:
                noisyData[i] = 1 - noisyData[i]
        return noisyData

    @staticmethod
    def gaussianFilter(data, std):
        shape = np.shape(data)
        noise = np.random.normal(0, std, shape)
        return np.add(data, noise)

    @staticmethod
    def sample(mu, sigma):
        random = Constants.getInstance().random
        epsilon = random.standard_normal()
        return epsilon, sigma * epsilon + mu