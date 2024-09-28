import os
import sys
import csv
import numpy as np
from .perceptron import Perceptron


class Utils:
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
