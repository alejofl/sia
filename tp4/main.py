import json
import os
import sys
import pandas as pd
from src.constants import Constants
from src.distance import DistanceCalculator
from src.kohonen import Kohonen


EUROPE_DATASET_PATH = os.path.join(os.path.dirname(sys.argv[0]), "resources", "europe.csv")


# -- KOHONEN ----------------------------------------------------------------
def solveKohonen(options):
    df = pd.read_csv(EUROPE_DATASET_PATH)
    dataset = df.to_numpy()

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

# -- END KOHONEN ------------------------------------------------------------


if __name__ == "__main__":
    with open(sys.argv[1], 'r') as configFile:
        config = json.load(configFile)

        Constants(seed=config["seed"])

        match config["problem"]:
            case "KOHONEN":
                solveKohonen(config["options"])

    sys.exit(0)
