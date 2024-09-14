# Genetic Algorithms - The Best Player for ITBUM ONLINE

## Overview

This project implements a series of genetic algorithms to select the best player for the ITBUM ONLINE game. The game is a role-playing game where the player can choose between different characters (warrior, archer, guardian and wizard), build their army and fight against other players. The goal of the project is to find the best player for the game by using genetic algorithms, given the character type and a maximum number of points that can be used to create the character.

## Features

The project implements the following features:

* **Selection Methods**: the user can divide the population into different groups and select the best individuals from each group, using different selection methods. Moreover, the user can select other divisions and selection methods for the replacement of the population, before starting the loop again. The following selection methods are implemented:
    * Elite
    * Roulette Wheel
    * Universal
    * Boltzmann
    * Ranking
    * Deterministic Tournament
    * Probabilistic Tournament

* **Crossover Methods**:
    * Single Point
    * Two Points
    * Anular
    * Uniform

* **Mutation Methods**:
    * Single-Gene Mutation (GEN)
    * Uniform Multi-Gene Mutation (MULTIGEN)

* **Replacement Methods**:
    * Traditional
    * Young-Biased

* **Stop Conditions**:
    * Execution time.
    * Maximum number of generations.
    * Maximum number of generations in which a percentage of the population has not changed.
    * Maximum number of generations in which the best individual has not changed.
    * Maximum number of generations in which the best individual has been inside a delta.

- **Performance Metrics**: After running the genetic algorithms, the engine outputs:
  - Characteristics of the best individual.
  - Fitness of the best individual.
  - Number of generations.
  - Execution time.
  - Number of created children.
  - Number of mutations.

## How to Run

### Prerequisites

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

Where `<config_file>` is the path to the configuration file, where all of the parameters for both the game and the genetic algorithm engine are stored. The configuration file must be a JSON file which validates against the `config/schema.json` JSON Schema file.

An example configuration file is provided in the `config` directory, as well as the JSON Schema file.

The engine will output a CSV file containing the performance metrics for each repetition.