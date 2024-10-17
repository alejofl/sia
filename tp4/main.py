import json
import sys
import pandas as pd
from src.constants import Constants
from src.distance import DistanceCalculator
from src.kohonen import Kohonen
from src.utils import Utils


# -- KOHONEN ----------------------------------------------------------------
def solveKohonen(options):
    countries, dataset = Utils.getEuropeDataset()

    kohonen = Kohonen(
        trainingSet=dataset,
        matrixSquareSize=options["matrixSquareSize"],
        maxEpochs=500 * len(dataset),
        initialLearningRate=options["initialLearningRate"],
        initialRadius=options["initialRadius"],
        changeLearningRate=options["changeLearningRate"],
        changeRadius=options["changeRadius"],
        distanceCalculator=DistanceCalculator.getCalculator(options["distanceCalculator"])
    )
    kohonen.train()
    
    for country, data in zip(countries, dataset):
        winnerNeuron = kohonen.test(data)
        print(f"{country} - {winnerNeuron}")

# -- END KOHONEN ------------------------------------------------------------


if __name__ == "__main__":
    with open(sys.argv[1], 'r') as configFile:
        config = json.load(configFile)

        Constants(seed=config["seed"])

        match config["problem"]:
            case "KOHONEN":
                solveKohonen(config["options"])

    sys.exit(0)
