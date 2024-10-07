# Supervised Learning - Single-Layer and Multilayer Peceptron

## Overview

This project implements a single-layer and multilayer perceptron to solve different problems. The perceptron is a type of artificial neural network that is used to classify input data or solve regression problems. The perceptron is trained using a supervised learning algorithm, which adjusts the weights of the network to minimize the error between the predicted output and the actual output.

## Features

The project implements the following features:

* **Activation Functions**: the user can choose between different activation functions for the neurons in the perceptron. The following activation functions are implemented:
    * Step
    * Linear
    * Logistic
    * Tanh

* **Optimization Algorithms**: the user can choose between different optimization algorithms to train the perceptron. The following optimization algorithms are implemented:
    * Gradient Descent
    * Momentum
    * Adam

* **Weights Updater**: the user can choose between different methods to update the weights of the perceptron. The following methods are implemented:
    * Batch
    * Mini-Batch
    * Online

* **Problems**: the user can choose between different problems to solve using the perceptron. The following problems are implemented:
    * AND (with STEP activation function)
    * XOR (with STEP activation function)
    * SET (with linear or non-linear activation function)
    * Multilayer XOR (with a multilayer perceptron)
    * Parity (with a multilayer perceptron)
    * Digits (with a multilayer perceptron)

* **Metrics**: After training the perceptron, the engine may output:
    * Mean Squared Error
    * Accuracy
    * Precision
    * Recall
    * F1 Score

## How to Run

### Prerequisites

The following libraries are required to run the code:

- Python 3
- Pipenv

You can install the required packages using:

```bash
pipenv install
```

### Running the Engine

To run the engine, execute the following command:

```bash
pipenv run python main.py <config_file>
```

Where `<config_file>` is the path to the configuration file, where all of the parameters for the engine are stored. The configuration file must be a JSON file with the following structure:

```json
{
    "epsilon": 1e-5,
    "maxEpochs": 1000,
    "seed": 732,
    "learning": {
        "optimizer": {
            "type": "GRADIENT_DESCENT | MOMENTUM | ADAM",
            "options": {
                "rate": 0.1,
                "alpha": 0.8, // Only needed for MOMENTUM
                "beta1": 0.9, // Only needed for ADAM
                "beta2": 0.999 // Only needed for ADAM
            }
        },
        "updater": {
            "type": "BATCH | MINI-BATCH | ONLINE",
            "options": {
                "updateEveryXInputs": 15 // Only needed for MINI-BATCH
            }
        }
    },
    "perceptron": {
        "problem": "AND | XOR | SET | MULTILAYER_XOR | PARITY | DIGITS",
        "architecture": [
            {
                "neuronQty": 1,
                "activationFunction": {
                    "type": "STEP | LINEAR | LOGISTIC | TANH",
                    "options": {
                        "beta": 1 // Only needed for LOGISTIC or TANH
                    }
                }
            }
        ]
    }
}
```

An example configuration file is provided in the `config` directory.

The engine will print the results of the testing phase.