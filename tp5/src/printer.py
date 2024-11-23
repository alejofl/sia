import numpy as np
import csv
import os
import sys
from .autoencoder import Autoencoder


class Printer:
    @staticmethod
    def mseVsEpoch(filename, autoencoder: Autoencoder):
        maxEpoch = len(autoencoder.layers[0][0].weightsPerEpoch)
        
        filepath = os.path.join(os.path.dirname(sys.argv[0]), "results", filename)
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w") as file:
            writer = csv.writer(file)
            writer.writerow(["epoch", "mse"])
            
            for i in range(maxEpoch):
                for layer in autoencoder.layers:
                    for neuron in layer:
                        neuron.weights = neuron.weightsPerEpoch[i]
                        
                mse = autoencoder.calculateError(autoencoder.inputs, autoencoder.expectedOutputs)
                writer.writerow([i, mse])
            
            for layer in autoencoder.layers:
                for neuron in layer:
                    neuron.weights = neuron.weightsPerEpoch[-1]

    @staticmethod
    def latentSpaceOutput(filename, autoencoder: Autoencoder, labels):
        filepath = os.path.join(os.path.dirname(sys.argv[0]), "results", filename)
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w") as file:
            writer = csv.writer(file)
            writer.writerow(["label", "x", "y"])

            for i, input in enumerate(autoencoder.inputs):
                latentSpaceOutput = autoencoder.getLatentSpaceOutput(input)
                writer.writerow([labels[i], *latentSpaceOutput[1:]])
