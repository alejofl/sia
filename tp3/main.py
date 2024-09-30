import sys
import json
import os
import pandas as pd
import numpy as np
from src.constants import Constants
from src.function import ActivationFunction
from src.perceptron import SingleLayerPerceptron, MultiLayerPerceptron
from src.utils import Utils


AND_DATASET_PATH = os.path.join(os.path.dirname(sys.argv[0]), "resources", "ej1", "and.csv")
def solveAnd(config):
    dataset = pd.read_csv(AND_DATASET_PATH)
    expectedOutputs = dataset["y"].to_numpy()
    inputs = dataset.drop(columns=["y"]).to_numpy() # TODO: see how to separate training set from testing set

    f = ActivationFunction.getFunction(config["perceptron"]["activationFunction"][0]["type"], config["perceptron"]["activationFunction"][0]["options"])
    w = Utils.initializeWeights(len(inputs[0]))
    p = SingleLayerPerceptron(f, w)
    p.train(inputs, expectedOutputs)

    print(p.test(np.array([1, -1, -1]))) # -1
    print(p.test(np.array([1, -1, 1]))) # -1
    print(p.test(np.array([1, 1, -1]))) # -1
    print(p.test(np.array([1, 1, 1]))) # 1


XOR_DATASET_PATH = os.path.join(os.path.dirname(sys.argv[0]), "resources", "ej1", "xor.csv")
def solveXor(config):
    dataset = pd.read_csv(XOR_DATASET_PATH)
    expectedOutputs = dataset["y"].to_numpy()
    inputs = dataset.drop(columns=["y"]).to_numpy() # TODO: see how to separate training set from testing set

    f = ActivationFunction.getFunction(config["perceptron"]["activationFunction"][0]["type"], config["perceptron"]["activationFunction"][0]["options"])
    w = Utils.initializeWeights(len(inputs[0]))
    p = SingleLayerPerceptron(f, w)
    p.train(inputs, expectedOutputs)

    print(p.test(np.array([1, -1, -1]))) # -1
    print(p.test(np.array([1, -1, 1]))) # 1
    print(p.test(np.array([1, 1, -1]))) # 1
    print(p.test(np.array([1, 1, 1]))) # -1


SET_DATASET_PATH = os.path.join(os.path.dirname(sys.argv[0]), "resources", "ej2", "set.csv")
def solveSet(config):
    dataset = pd.read_csv(SET_DATASET_PATH)
    testingSet = dataset.sample(frac=0.2) # TODO: change this to k cross validation
    trainingSet = dataset.drop(testingSet.index)

    trainingExpectedOutputs = trainingSet["y"].to_numpy()
    trainingInputs = trainingSet.drop(columns=["y"]).to_numpy() 
    f = ActivationFunction.getFunction(config["perceptron"]["activationFunction"][0]["type"], config["perceptron"]["activationFunction"][0]["options"])
    w = Utils.initializeWeights(len(trainingInputs[0]))
    p = SingleLayerPerceptron(f, w)
    p.train(trainingInputs, trainingExpectedOutputs)

    testingExpectedOutputs = testingSet["y"].to_numpy()
    testingInputs = testingSet.drop(columns=["y"]).to_numpy()
    for input, expectedOutput in zip(testingInputs, testingExpectedOutputs):
        output = p.test(input)
        print(output, expectedOutput, np.abs(output - expectedOutput) <= Constants.getInstance().epsilon, sep=" ")
    Utils.mseVsEpoch(p, trainingInputs, trainingExpectedOutputs, testingInputs, testingExpectedOutputs, "setMSE.csv")


if __name__ == "__main__":
    with open(sys.argv[1], 'r') as configFile:
        config = json.load(configFile)

        updateWeightsEveryXInputs = 1
        if config["learning"]["type"] == "BATCH":
            updateWeightsEveryXInputs = -1
        elif config["learning"]["type"] == "MINI-BATCH":
            updateWeightsEveryXInputs = config["learning"]["options"]["updateEveryXInputs"]
            
        Constants(
            epsilon=config["epsilon"],
            seed=config["seed"],
            maxEpochs=config["maxEpochs"],
            learningRate=config["learning"]["rate"],
            updateWeightsEveryXInputs=updateWeightsEveryXInputs
        )

        match config["perceptron"]["problem"]:
            case "AND":
                solveAnd(config)
            case "XOR":
                solveXor(config)
            case "SET":
                solveSet(config)
            case "PARITY":
                pass
            case "DIGITS":
                pass

    sys.exit(0)
