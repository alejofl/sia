import os
import sys
import csv
import numpy as np
from sklearn.model_selection import KFold, ShuffleSplit
from scipy.ndimage import gaussian_filter
from .function import ActivationFunction
from .perceptron import Perceptron, MultiLayerPerceptron
from .constants import Constants
from .metrics import Metrics


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
    def mseVsEpochToList(perceptron: Perceptron, trainingInputs, trainingExpectedOutputs, testingInputs, testingExpectedOutputs):
        mse = []
        for i, weights in enumerate(perceptron.weightsPerEpoch):
            perceptron.weights = weights
            trainingMSE = perceptron.calculateError(trainingInputs, trainingExpectedOutputs)
            testingMSE = perceptron.calculateError(testingInputs, testingExpectedOutputs)
            mse.append((trainingMSE, testingMSE))
        perceptron.weights = perceptron.weightsHistory[-1]
        return mse

    @staticmethod
    def meanMseVsEpoch(mse, filename):
        filepath = os.path.join(os.path.dirname(sys.argv[0]), "results", filename)
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w") as file:
            writer = csv.DictWriter(file, fieldnames=["epoch", "trainingMSE", "stdTrainingMSE", "testingMSE", "stdTestingMSE"])
            writer.writeheader()
            for i in range(len(mse[0])):
                trainingMSE = np.mean([x[i][0] for x in mse])
                stdTrainingMSE = np.std([x[i][0] for x in mse])
                testingMSE = np.mean([x[i][1] for x in mse])
                stdTestingMSE = np.std([x[i][1] for x in mse])
                writer.writerow({"epoch": i, "trainingMSE": trainingMSE, "stdTrainingMSE": stdTrainingMSE, "testingMSE": testingMSE, "stdTestingMSE": stdTestingMSE})

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
    def getArbitrarySets(inputs, expectedOutputs, trainingPercentage):
        inputs = np.array(inputs)
        expectedOutputs = np.array(expectedOutputs)
        trainingIndices = Constants.getInstance().random.choice(len(inputs), np.floor(len(inputs) * trainingPercentage).astype(int), replace=False)
        trainingInputs = inputs[trainingIndices]
        trainingExpectedOutputs = expectedOutputs[trainingIndices]
        testingInputs = np.delete(inputs, trainingIndices, axis=0)
        testingExpectedOutputs = np.delete(expectedOutputs, trainingIndices, axis=0)
        return [(trainingInputs, trainingExpectedOutputs, testingInputs, testingExpectedOutputs)]

    @staticmethod
    def getKFoldCrossValidationSets(inputs, expectedOutputs, k):
        inputs = np.array(inputs)
        expectedOutputs = np.array(expectedOutputs)
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
        inputs = np.array(inputs)
        expectedOutputs = np.array(expectedOutputs)
        rs = ShuffleSplit(n_splits=k, test_size=0.2, random_state=Constants.getInstance().seed)
        datasets = []
        for trainingIndexes, testingIndexes in rs.split(inputs, expectedOutputs):
            trainingInputs = inputs[trainingIndexes]
            trainingExpectedOutputs = expectedOutputs[trainingIndexes]
            testingInputs = inputs[testingIndexes]
            testingExpectedOutputs = expectedOutputs[testingIndexes]
            datasets.append((trainingInputs, trainingExpectedOutputs, testingInputs, testingExpectedOutputs))
        return datasets
    
    @staticmethod
    def digits_accuracyVsEpoch(perceptron: MultiLayerPerceptron, trainingInputs, trainingExpectedOutputs, testingInputs, testingExpectedOutputs, filename):
        filepath = os.path.join(os.path.dirname(sys.argv[0]), "results", filename)
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w") as file:
            writer = csv.DictWriter(file, fieldnames=["epoch", "trainingAccuracy", "testingAccuracy"])
            writer.writeheader()
            for epoch in range(len(perceptron.layers[0][0].getWeightsPerEpoch())):
                for layer in perceptron.layers:
                    for neuron in layer:
                        neuron.weights = neuron.getWeightsPerEpoch()[epoch]
                
                trainingPredicted = np.array([np.argmax(perceptron.test(input)) for input in trainingInputs])
                trainingExpected = np.array([np.argmax(expected) for expected in trainingExpectedOutputs])
                testingPredicted = np.array([np.argmax(perceptron.test(input)) for input in testingInputs])
                testingExpected = np.array([np.argmax(expected) for expected in testingExpectedOutputs])
                trainingMetrics = Metrics(trainingExpected, trainingPredicted)
                testingMetrics = Metrics(testingExpected, testingPredicted)
                writer.writerow({"epoch": epoch, "trainingAccuracy": trainingMetrics.accuracy(), "testingAccuracy": testingMetrics.accuracy()})
            for layer in perceptron.layers:
                for neuron in layer:
                    neuron.weights = neuron.weightsHistory[-1]
                    
    @staticmethod
    def parity_accuracyVsEpoch(perceptron: MultiLayerPerceptron, trainingInputs, trainingExpectedOutputs, testingInputs, testingExpectedOutputs, filename):
        filepath = os.path.join(os.path.dirname(sys.argv[0]), "results", filename)
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w") as file:
            writer = csv.DictWriter(file, fieldnames=["epoch", "trainingAccuracy", "testingAccuracy"])
            writer.writeheader()
            for epoch in range(len(perceptron.layers[0][0].getWeightsPerEpoch())):
                for layer in perceptron.layers:
                    for neuron in layer:
                        neuron.weights = neuron.getWeightsPerEpoch()[epoch]
                
                trainingPredicted = np.array([perceptron.test(input)[0] for input in trainingInputs])
                trainingExpected = np.array([expected[0] for expected in trainingExpectedOutputs])
                testingPredicted = np.array([perceptron.test(input)[0] for input in testingInputs])
                testingExpected = np.array([expected[0] for expected in testingExpectedOutputs])
                trainingMetrics = Metrics(trainingExpected, trainingPredicted)
                testingMetrics = Metrics(testingExpected, testingPredicted)
                writer.writerow({"epoch": epoch, "trainingAccuracy": trainingMetrics.accuracy(), "testingAccuracy": testingMetrics.accuracy()})
            for layer in perceptron.layers:
                for neuron in layer:
                    neuron.weights = neuron.weightsHistory[-1]
                    
    @staticmethod
    def parity_accuracyVsEpochToList(perceptron: MultiLayerPerceptron, trainingInputs, trainingExpectedOutputs, testingInputs, testingExpectedOutputs):
        accuracy = []
        for epoch in range(len(perceptron.layers[0][0].getWeightsPerEpoch())):
            for layer in perceptron.layers:
                for neuron in layer:
                    neuron.weights = neuron.getWeightsPerEpoch()[epoch]
            
            trainingPredicted = np.array([perceptron.test(input)[0] for input in trainingInputs])
            trainingExpected = np.array([expected[0] for expected in trainingExpectedOutputs])
            testingPredicted = np.array([perceptron.test(input)[0] for input in testingInputs])
            testingExpected = np.array([expected[0] for expected in testingExpectedOutputs])
            trainingMetrics = Metrics(trainingExpected, trainingPredicted)
            testingMetrics = Metrics(testingExpected, testingPredicted)
            accuracy.append((trainingMetrics.accuracy(), testingMetrics.accuracy()))
        for layer in perceptron.layers:
            for neuron in layer:
                neuron.weights = neuron.weightsHistory[-1]
        return accuracy
    
    @staticmethod
    def meanAccuracyVsEpoch(accuracy, filename):
        filepath = os.path.join(os.path.dirname(sys.argv[0]), "results", filename)
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w") as file:
            writer = csv.DictWriter(file, fieldnames=["epoch", "trainingAccuracy", "stdTrainingAccuracy", "testingAccuracy", "stdTestingAccuracy"])
            writer.writeheader()
            for i in range(len(accuracy[0])):
                trainingAccuracy = np.mean([x[i][0] for x in accuracy])
                stdTrainingAccuracy = np.std([x[i][0] for x in accuracy])
                testingAccuracy = np.mean([x[i][1] for x in accuracy])
                stdTestingAccuracy = np.std([x[i][1] for x in accuracy])
                writer.writerow({"epoch": i, "trainingAccuracy": trainingAccuracy, "stdTrainingAccuracy": stdTrainingAccuracy, "testingAccuracy": testingAccuracy, "stdTestingAccuracy": stdTestingAccuracy})
