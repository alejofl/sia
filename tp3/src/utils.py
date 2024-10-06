import os
import sys
import csv
import numpy as np
from sklearn.model_selection import KFold, ShuffleSplit
from scipy.ndimage import gaussian_filter
from .function import ActivationFunction
from .perceptron import Perceptron
from .constants import Constants


class Utils:
    @staticmethod
    def initializeWeights(length, zeros=False):
        if zeros:
            return np.zeros(length)
        r = Constants.getInstance().random
        return r.uniform(-1, 1, length)

    @staticmethod
    def mseVsEpoch(perceptron: Perceptron, trainingInputs, trainingExpectedOutputs, testingInputs, testingExpectedOutputs, filename):
        filepath = os.path.join(os.path.dirname(sys.argv[0]), "results", filename)
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w") as file:
            writer = csv.DictWriter(file, fieldnames=["epoch", "trainingMSE", "testingMSE"])
            writer.writeheader()
            for i, weights in enumerate(perceptron.weightsPerEpoch):
                perceptron.weights = weights
                trainingMSE = perceptron.calculateError(trainingInputs, trainingExpectedOutputs)
                testingMSE = perceptron.calculateError(testingInputs, testingExpectedOutputs)
                writer.writerow({"epoch": i, "trainingMSE": trainingMSE, "testingMSE": testingMSE})
            perceptron.weights = perceptron.weightsHistory[-1]

    @staticmethod
    def getMultilayerArchitecture(config, inputLength):
        architecture = []
        for i, layer in enumerate(config["perceptron"]["architecture"]):
            neuronQty = layer["neuronQty"]
            f = ActivationFunction.getFunction(layer["activationFunction"]["type"], layer["activationFunction"]["options"])
            weights = []
            for _ in range(neuronQty):
                weightsQty = config["perceptron"]["architecture"][i-1]["neuronQty"]+1 if i > 0 else inputLength
                weights.append(Utils.initializeWeights(weightsQty))
            architecture.append((neuronQty, f, weights))
        return architecture

    @staticmethod
    def getDigitRepresentation(filepath):
        with open(filepath, 'r') as file:
            representation = file.readlines()
            digits = []
            currentInput = np.zeros((7,5))
            for i in range(len(representation)):
                if i != 0 and i % 7 == 0:
                    digits.append(currentInput)
                    currentInput = np.zeros((7,5))
                row = representation[i][:-2].split(" ")
                for j in range(len(row)):
                    currentInput[i % 7][j] = 1 if row[j] == "1" else 0
            digits.append(currentInput)
            return digits

    @staticmethod
    def getPerceptronInputFromDigits(digits, sigma=0):
        inputs = []
        for digit in digits:
            newDigit = np.array([Constants.getInstance().bias])
            blurredDigit = gaussian_filter(digit, sigma=sigma)
            inputs.append(np.append(newDigit, blurredDigit.flatten()))
        return inputs

    @staticmethod
    def getKFoldCrossValidationSets(inputs, expectedOutputs, k):
        kf = KFold(n_splits=k)
        datasets = []
        for trainingIndexes, testingIndexes in kf.split(inputs, expectedOutputs):
            trainingInputs = inputs[trainingIndexes]
            trainingExpectedOutputs = expectedOutputs[trainingIndexes]
            testingInputs = inputs[testingIndexes]
            testingExpectedOutputs = expectedOutputs[testingIndexes]
            datasets.append((trainingInputs, trainingExpectedOutputs, testingInputs, testingExpectedOutputs))
        return datasets
    
    @staticmethod
    def getShuffleSplitSets(inputs, expectedOutputs, k):
        rs = ShuffleSplit(n_splits=k, test_size=0.2, random_state=Constants.getInstance().seed)
        datasets = []
        for trainingIndexes, testingIndexes in rs.split(inputs, expectedOutputs):
            trainingInputs = inputs[trainingIndexes]
            trainingExpectedOutputs = expectedOutputs[trainingIndexes]
            testingInputs = inputs[testingIndexes]
            testingExpectedOutputs = expectedOutputs[testingIndexes]
            datasets.append((trainingInputs, trainingExpectedOutputs, testingInputs, testingExpectedOutputs))
        return datasets
