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

    predicted = [p.test(np.array([1, 1, 1])), p.test(np.array([1, -1, 1])), p.test(np.array([1, 1, -1])), p.test(np.array([1, -1, -1]))]

    print("Predicted:")
    print(predicted[0]) # 1
    print(predicted[1]) # -1
    print(predicted[2]) # -1
    print(predicted[3]) # -1


    metrics = Metrics(expectedOutputs, predicted)
    
    print("Metrics:")
    print("accuracy", metrics.accuracy())
    print("precision", metrics.precision())
    print("recall", metrics.recall())

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

    print(p.test(np.array([1, -1, -1]))) # -1
    print(p.test(np.array([1, -1, 1]))) # 1
    print(p.test(np.array([1, 1, -1]))) # 1
    print(p.test(np.array([1, 1, 1]))) # -1

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
    for input, expectedOutput in zip(testingInputs, testingExpectedOutputs):
        output = p.test(input)
        print(output, expectedOutput, np.abs(output - expectedOutput) <= Constants.getInstance().epsilon, sep=" ")
    Utils.mseVsEpoch(p, trainingInputs, trainingExpectedOutputs, testingInputs, testingExpectedOutputs, "setMSE.csv")

# Exercise 3

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
            weightsQty = config["perceptron"]["architecture"][i-1]["neuronQty"] if i > 0 else len(inputs[0])
            weights.append(Utils.initializeWeights(weightsQty))
        architecture.append((neuronQty, f, weights))

    o = OptimizerFunction.getFunction(config["learning"]["optimizer"]["type"])
    p = MultiLayerPerceptron(architecture, o, config["learning"]["optimizer"]["options"])

    p.train(inputs, expectedOutputs)

    print(p.test(np.array([1, -1, -1]))) # -1
    print(p.test(np.array([1, -1, 1]))) # 1
    print(p.test(np.array([1, 1, -1]))) # 1
    print(p.test(np.array([1, 1, 1]))) # -1


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
                pass

    sys.exit(0)
