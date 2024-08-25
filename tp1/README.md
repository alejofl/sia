# Search Algorithms - Sokoban Solver

## Overview

This repository contains an implementation of a search engine designed to solve the **Sokoban** puzzle using various search methods. The goal is to find the most efficient solution in terms of the number of moves required to complete the puzzle. The implemented search algorithms include:

- Breadth-First Search (BFS)
- Depth-First Search (DFS)
- Greedy Search
- A*

## Features

- **State Representation**: The game state is represented in a manner suitable for efficient search and manipulation, allowing the search engine to traverse different states effectively.

- **Heuristics**:
  - Admissible Heuristics: Implemented to ensure optimal solutions. In particular, Manhattan Distance and Euclidean Distance are implemented.
  - Non-Admissible Heuristics: Optional, for experimental purposes. The engine features the Bounding Box heuristic.

- **Performance Metrics**: After solving the puzzle, the engine outputs:
  - Cost of the solution in terms of moves.
  - Number of expanded nodes.
  - Number of border nodes.
  - Solution path from the initial state to the goal state.
  - Execution time.

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

Where `<config_file>` is the path to the configuration file. The configuration file must be a JSON file which contains the following information:

- `board`: The path to the file containing the Sokoban puzzle.
- `algorithm`: The search algorithm to use. Possible values are `BFS`, `DFS`, `GREEDY`, and `A*`.
- `heuristic`: The heuristic to use. Possible values are `MANHATTAN`, `EUCLIDEAN`, and `BOUNDING_BOX`.
- `animation`: Whether to display the solution as an animation. Possible values are `GIF`, `ASCII` and `NONE`.
- `reps`: The number of times to repeat the search. This is useful for statistical purposes.

An example configuration file is provided in the `config` directory.

The engine will output a CSV file containing the performance metrics for each repetition. Additionally, it will display the animation with the solution path if requested.
