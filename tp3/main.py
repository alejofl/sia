import sys
import json
import os
import pandas as pd
import numpy as np
from src.constants import Constants
from src.function import ActivationFunction
from src.optimizer import OptimizerFunction
from src.perceptron import SingleLayerPerceptron, MultiLayerPerceptron
from src.utils import Utils
from src.metrics import Metrics

constants = None

# --------------------------------------------------------------------------------------------------------
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

    predicted = [p.test(np.array([constants.bias, 1, 1])), p.test(np.array([constants.bias, -1, 1])), p.test(np.array([constants.bias, 1, -1])), p.test(np.array([constants.bias, -1, -1]))]

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

    predicted = [p.test(np.array([constants.bias, -1, -1])), p.test(np.array([constants.bias, -1, 1])), p.test(np.array([constants.bias, 1, -1])), p.test(np.array([constants.bias, 1, 1]))]

    print(predicted[0]) # -1
    print(predicted[1]) # 1
    print(predicted[2]) # 1
    print(predicted[3]) # -1

    metrics = Metrics(expectedOutputs, predicted)
    metrics.displayAll()

# --------------------------------------------------------------------------------------------------------
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

    # TODO: decide if we want to show metrics

# --------------------------------------------------------------------------------------------------------
# Exercise 3 - Xor

def solveMultilayerXor(config):
    dataset = pd.read_csv(XOR_DATASET_PATH)
    expectedOutputs = []
    for o in dataset["y"].to_numpy():
        expectedOutputs.append([o])
    inputs = dataset.drop(columns=["y"]).to_numpy() # TODO: see how to separate training set from testing set

    architecture = Utils.getMultilayerArchitecture(config, len(inputs[0]))
    o = OptimizerFunction.getFunction(config["learning"]["optimizer"]["type"])
    p = MultiLayerPerceptron(architecture, o, config["learning"]["optimizer"]["options"])

    p.train(inputs, expectedOutputs)

    predicted = [p.test(np.array([constants.bias, -1, -1])), p.test(np.array([constants.bias, -1, 1])), p.test(np.array([constants.bias, 1, -1])), p.test(np.array([constants.bias, 1, 1]))]

    print(predicted[0]) # -1
    print(predicted[1]) # 1
    print(predicted[2]) # 1
    print(predicted[3]) # -1

    metrics = Metrics(expectedOutputs, predicted)
    metrics.displayAll()
    
# --------------------------------------------------------------------------------------------------------
# Exercise 3 - Digits

DIGITS_REPRESENTATION_PATH = os.path.join(os.path.dirname(sys.argv[0]), "resources", "ej3", "digits.txt")
DIGITS_DATASET_PATH = os.path.join(os.path.dirname(sys.argv[0]), "resources", "ej3", "digits.csv")
def solveDigits(config):
    digits = Utils.getDigitRepresentation(DIGITS_REPRESENTATION_PATH)
    dataset = pd.read_csv(DIGITS_DATASET_PATH)
    expectedOutputs = []
    for o in dataset["y"].to_numpy():
        outputRow = []
        for i in range(10):
            outputRow.append(1 if i == o else 0)
        expectedOutputs.append(outputRow)

    inputs = Utils.getPerceptronInputFromDigits(digits)
    
    architecture = Utils.getMultilayerArchitecture(config, len(inputs[0]))
    o = OptimizerFunction.getFunction(config["learning"]["optimizer"]["type"])
    p = MultiLayerPerceptron(architecture, o, config["learning"]["optimizer"]["options"])
    p.train(inputs, expectedOutputs)

    testingInputs = Utils.getPerceptronInputFromDigits(digits, 0.4)
    predicted = []
    for digit in testingInputs:
        x = p.test(digit)
        print(x)
        # predicted.append(np.argmax(x))

    # metrics = Metrics([0,1,2,3,4,5,6,7,8,9], predicted)
    # metrics.displayAll()
    
# --------------------------------------------------------------------------------------------------------
# Exercise 3 - Parity

PARITY_DATASET_PATH = os.path.join(os.path.dirname(sys.argv[0]), "resources", "ej3", "parity.csv")
def solveParity(config):
    digits = Utils.getDigitRepresentation(DIGITS_REPRESENTATION_PATH)
    dataset = pd.read_csv(PARITY_DATASET_PATH)
    expectedOutputs = []
    for o in dataset["y"].to_numpy():
        expectedOutputs.append([o])

    inputs = Utils.getPerceptronInputFromDigits(digits)
    
    architecture = Utils.getMultilayerArchitecture(config, len(inputs[0]))
    o = OptimizerFunction.getFunction(config["learning"]["optimizer"]["type"])
    p = MultiLayerPerceptron(architecture, o, config["learning"]["optimizer"]["options"])
    p.train(inputs, expectedOutputs)

    testingInputs = Utils.getPerceptronInputFromDigits(digits, 0.4)
    predicted = []
    for digit in testingInputs:
        x = p.test(digit)
        print(x)
        # predicted.append(np.argmax(x))

    # metrics = Metrics(dataset["y"].to_numpy(), predicted)
    # metrics.displayAll()


if __name__ == "__main__":
    with open(sys.argv[1], 'r') as configFile:
        config = json.load(configFile)

        updateWeightsEveryXInputs = 1
        if config["learning"]["updater"]["type"] == "BATCH":
            updateWeightsEveryXInputs = -1
        elif config["learning"]["updater"]["type"] == "MINI-BATCH":
            updateWeightsEveryXInputs = config["learning"]["updater"]["options"]["updateEveryXInputs"]
            
        constants = Constants(
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
                solveParity(config)
            case "DIGITS":
                solveDigits(config)

    sys.exit(0)
