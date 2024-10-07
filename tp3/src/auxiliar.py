import sys
import csv
import os
import pandas as pd
import numpy as np
from .function import ActivationFunction
from .optimizer import OptimizerFunction
from .perceptron import SingleLayerPerceptron
from .utils import Utils


class Auxiliar:
    @staticmethod
    def ex2_studyBeta(config):
        filepath = os.path.join(os.path.dirname(sys.argv[0]), "results", "ej2_betas_tanh.csv")
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        SET_DATASET_PATH = os.path.join(os.path.dirname(sys.argv[0]), "resources", "ej2", "set.csv")
        dataset = pd.read_csv(SET_DATASET_PATH)
        expectedOutputs = dataset["y"].to_numpy()
        inputs = dataset.drop(columns=["y"]).to_numpy()

        betas = [0.1, 0.4, 0.7, 1, 2.5, 5, 10, 20, 50, 100]
        
        with open(filepath, "w") as file:
            writer = csv.DictWriter(file, fieldnames=["beta", "mse"])
            writer.writeheader()

            o = OptimizerFunction.getFunction(config["learning"]["optimizer"]["type"])(config["learning"]["optimizer"]["options"])
            for beta in betas:
                print("Training with beta =", beta)
                
                f = ActivationFunction.getFunction(config["perceptron"]["architecture"][0]["activationFunction"]["type"], {"beta": beta})
                f.configureNormalization(np.min(expectedOutputs), np.max(expectedOutputs))
                w = Utils.initializeWeights(len(inputs[0]))
                p = SingleLayerPerceptron(f, o, w)
                p.train(inputs, expectedOutputs)
                
                writer.writerow({"beta": beta, "mse": p.calculateError(inputs, expectedOutputs)})
                
                print("Finished training with beta =", beta)
