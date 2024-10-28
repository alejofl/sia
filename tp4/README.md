# Unsupervised Learning - Kohonen Self Organizing Maps, PCA, Oja Rule & Hopfield Networks

## Overview

This repository contains the implementation of a few unsupervised learning algorithms. When we talk about unsupervised learning, we are talking about a type of machine learning that looks for previously undetected patterns in a data set with no pre-existing labels and with a minimum of human supervision. In this case, we implemented algorithms with focus on associative memory and dimensionality reduction.

## Features

The implemented algorithms are:

* **Kohonen Self Organizing Maps:** This algorithm is used to reduce the dimensionality of the input space and to group similar data points together. In particular, we will group similar Europe countries based on a dataset with seven economic and social variables.

* **Oja Rule:** This algorithm is used to reduce the dimensionality of the input space. The Oja Rule is used to calculate the first principal component iteratively. In particular, we will reduce the dimensionality of the Europe dataset mentioned above, and compare the results with the Kohonen Self Organizing Maps and PCA algorithm.

* **Hopfield Networks:** This algorithm is used to recognise offuscated patterns from a group of previously stored patterns. In particular, we will store and retrieve letters stored in a 5x5 matrix. We will also test the algorithm with a noisy version of the letters.

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
    "problem": "KOHONEN | OJA | HOPFIELD | HOPFIELD - ORTOGONALITY",
    "options": {
	    "matrixSquareSize": 3, // Only for KOHONEN
	    "initialLearningRate": 0.3, // Only for KOHONEN or OJA
	    "initialRadius": 1, // Only for KOHONEN
	    "changeLearningRate": false, // Only for KOHONEN or OJA
	    "changeRadius": false, // Only for KOHONEN
	    "distanceCalculator": "EUCLIDEAN", // Only for KOHONEN
        "maxEpochs": 10000, // Only for OJA or HOPFIELD
        "noiseLevel": 0.1, // Only for HOPFIELD
        "letters": ["F", "I", "O", "V"] // Only for HOPFIELD
    },
    "seed": 732
}
```

An example configuration file is provided in the `config` directory.

The engine will output a CSV file or print to screen the solution to the problem.