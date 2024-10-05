import sys
import json
import os
import pandas as pd
import numpy as np
from scipy.ndimage import gaussian_filter
from src.constants import Constants
from src.function import ActivationFunction
from src.optimizer import OptimizerFunction
from src.perceptron import SingleLayerPerceptron, MultiLayerPerceptron
from src.utils import Utils
from src.metrics import Metrics

BIAS = 1

# Exercise 1

AND_DATASET_PATH = os.path.join(os.path.dirname(sys.argv[0]), "resources", "ej1", "and.csv")
def solveAnd(config):
    dataset = pd.read_csv(AND_DATASET_PATH)
    expectedOutputs = dataset["y"].to_numpy()
    inputs = dataset.drop(columns=["y"]).to_numpy() # TODO: see how to separate training set from testing set

    f = ActivationFunction.getFunction(config["perceptron"]["architecture"][0]["activationFunction"]["type"], config["perceptron"]["architecture"][0]["activationFunction"]["options"])
    o = OptimizerFunction.getFunction(config["learning"]["optimizer"]["type"])(config["learning"]["optimizer"]["options"])
    w = Utils.initializeWeights(len(inputs[0]))
    p = SingleLayerPerceptron(f, o, w)
    p.train(inputs, expectedOutputs)

    predicted = [p.test(np.array([BIAS, 1, 1])), p.test(np.array([BIAS, -1, 1])), p.test(np.array([BIAS, 1, -1])), p.test(np.array([BIAS, -1, -1]))]

    print("Predicted:")
    print(predicted[0]) # 1
    print(predicted[1]) # -1
    print(predicted[2]) # -1
    print(predicted[3]) # -1

    metrics = Metrics(expectedOutputs, predicted)
    metrics.displayAll()
    

XOR_DATASET_PATH = os.path.join(os.path.dirname(sys.argv[0]), "resources", "ej1", "xor.csv")
def solveXor(config):
    dataset = pd.read_csv(XOR_DATASET_PATH)
    expectedOutputs = dataset["y"].to_numpy()
    inputs = dataset.drop(columns=["y"]).to_numpy() # TODO: see how to separate training set from testing set

    f = ActivationFunction.getFunction(config["perceptron"]["architecture"][0]["activationFunction"]["type"], config["perceptron"]["architecture"][0]["activationFunction"]["options"])
    o = OptimizerFunction.getFunction(config["learning"]["optimizer"]["type"])(config["learning"]["optimizer"]["options"])
    w = Utils.initializeWeights(len(inputs[0]))
    p = SingleLayerPerceptron(f, o, w)
    p.train(inputs, expectedOutputs)

    predicted = [p.test(np.array([BIAS, -1, -1])), p.test(np.array([BIAS, -1, 1])), p.test(np.array([BIAS, 1, -1])), p.test(np.array([BIAS, 1, 1]))]

    print(predicted[0]) # -1
    print(predicted[1]) # 1
    print(predicted[2]) # 1
    print(predicted[3]) # -1

    metrics = Metrics(expectedOutputs, predicted)
    metrics.displayAll()

# Exercise 2

SET_DATASET_PATH = os.path.join(os.path.dirname(sys.argv[0]), "resources", "ej2", "set.csv")
def solveSet(config):
    dataset = pd.read_csv(SET_DATASET_PATH)
    testingSet = dataset.sample(frac=0.2) # TODO: change this to k cross validation
    trainingSet = dataset.drop(testingSet.index)

    trainingExpectedOutputs = trainingSet["y"].to_numpy()
    trainingInputs = trainingSet.drop(columns=["y"]).to_numpy() 
    f = ActivationFunction.getFunction(config["perceptron"]["architecture"][0]["activationFunction"]["type"], config["perceptron"]["architecture"][0]["activationFunction"]["options"])
    o = OptimizerFunction.getFunction(config["learning"]["optimizer"]["type"])(config["learning"]["optimizer"]["options"])
    w = Utils.initializeWeights(len(trainingInputs[0]))
    p = SingleLayerPerceptron(f, o, w)
    p.train(trainingInputs, trainingExpectedOutputs)

    testingExpectedOutputs = testingSet["y"].to_numpy()
    testingInputs = testingSet.drop(columns=["y"]).to_numpy()
    testingOutputs = []
    for input, expectedOutput in zip(testingInputs, testingExpectedOutputs):
        output = p.test(input)
        testingOutputs.append(output)
        print(output, expectedOutput, np.abs(output - expectedOutput) <= Constants.getInstance().epsilon, sep=" ")
    Utils.mseVsEpoch(p, trainingInputs, trainingExpectedOutputs, testingInputs, testingExpectedOutputs, "setMSE.csv")

    #TODO: decide if we want to show metrics

# Exercise 3 - Xor

def solveMultilayerXor(config):
    dataset = pd.read_csv(XOR_DATASET_PATH)
    tmp = dataset["y"].to_numpy()
    expectedOutputs = []
    for o in tmp:
        expectedOutputs.append([o])
    inputs = dataset.drop(columns=["y"]).to_numpy() # TODO: see how to separate training set from testing set

    architecture = []
    for i, layer in enumerate(config["perceptron"]["architecture"]):
        neuronQty = layer["neuronQty"]
        f = ActivationFunction.getFunction(layer["activationFunction"]["type"], layer["activationFunction"]["options"])
        weights = []
        for _ in range(neuronQty):
            weightsQty = config["perceptron"]["architecture"][i-1]["neuronQty"]+1 if i > 0 else len(inputs[0])
            weights.append(Utils.initializeWeights(weightsQty))
        architecture.append((neuronQty, f, weights))

    o = OptimizerFunction.getFunction(config["learning"]["optimizer"]["type"])
    p = MultiLayerPerceptron(architecture, o, config["learning"]["optimizer"]["options"])

    p.train(inputs, expectedOutputs)

    predicted = [p.test(np.array([BIAS, -1, -1])), p.test(np.array([BIAS, -1, 1])), p.test(np.array([BIAS, 1, -1])), p.test(np.array([BIAS, 1, 1]))]

    print(predicted[0]) # -1
    print(predicted[1]) # 1
    print(predicted[2]) # 1
    print(predicted[3]) # -1

    metrics = Metrics(expectedOutputs, predicted)
    metrics.displayAll()
    
# Exercise 3 - Digits

DIGITS_REPRESENTATION_PATH = os.path.join(os.path.dirname(sys.argv[0]), "resources", "ej3", "digits.txt")
DIGITS_DATASET_PATH = os.path.join(os.path.dirname(sys.argv[0]), "resources", "ej3", "digits.csv")
def solveDigits(config):
    representation = []
    with open(DIGITS_REPRESENTATION_PATH, 'r') as file:
        representation = file.readlines()
    dataset = pd.read_csv(DIGITS_DATASET_PATH)
    expectedOutputs = []
    for o in dataset["y"].to_numpy():
        outputRow = []
        for i in range(10):
            outputRow.append(1 if i == o else 0)
        expectedOutputs.append(outputRow)

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
    
    blurred = []
    for digit in digits:
        blurred.append(gaussian_filter(digit, sigma=0.4))
    
    inputs = []
    for digit in digits:
        newDigit = np.array([BIAS])
        inputs.append(np.append(newDigit, digit.flatten()))
    
    architecture = []
    for i, layer in enumerate(config["perceptron"]["architecture"]):
        neuronQty = layer["neuronQty"]
        f = ActivationFunction.getFunction(layer["activationFunction"]["type"], layer["activationFunction"]["options"])
        weights = []
        for _ in range(neuronQty):
            weightsQty = config["perceptron"]["architecture"][i-1]["neuronQty"]+1 if i > 0 else len(inputs[0])
            weights.append(Utils.initializeWeights(weightsQty))
        architecture.append((neuronQty, f, weights))

    o = OptimizerFunction.getFunction(config["learning"]["optimizer"]["type"])
    p = MultiLayerPerceptron(architecture, o, config["learning"]["optimizer"]["options"])
    p.train(inputs, expectedOutputs)

    tests = []
    for digit in blurred:
        newDigit = np.array([BIAS])
        tests.append(np.append(newDigit, digit.flatten()))
    predicted = []
    for digit in tests:
        x = p.test(digit)
        print(x)
        predicted.append(np.argmax(x))

    metrics = Metrics([0,1,2,3,4,5,6,7,8,9], predicted)
    metrics.displayAll()

    # Utils.mseVsEpoch(p, inputs, expectedOutputs, inputs, expectedOutputs, "digitsMSE.csv")


if __name__ == "__main__":
    with open(sys.argv[1], 'r') as configFile:
        config = json.load(configFile)

        updateWeightsEveryXInputs = 1
        if config["learning"]["updater"]["type"] == "BATCH":
            updateWeightsEveryXInputs = -1
        elif config["learning"]["updater"]["type"] == "MINI-BATCH":
            updateWeightsEveryXInputs = config["learning"]["updater"]["options"]["updateEveryXInputs"]
            
        Constants(
            epsilon=config["epsilon"],
            seed=config["seed"],
            maxEpochs=config["maxEpochs"],
            updateWeightsEveryXInputs=updateWeightsEveryXInputs
        )

        match config["perceptron"]["problem"]:
            case "AND":
                solveAnd(config)
            case "XOR":
                solveXor(config)
            case "SET":
                solveSet(config)
            case "MULTILAYER_XOR":
                solveMultilayerXor(config)
            case "PARITY":
                pass
            case "DIGITS":
                solveDigits(config)

    sys.exit(0)
