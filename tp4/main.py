import json
import sys
import pandas as pd
from src.constants import Constants
from src.distance import DistanceCalculator
from src.kohonen import Kohonen
from src.oja import Oja
from src.hopfield import Hopfield
from src.utils import Utils
from src.plotter import Plotter


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
    
    #for country, data in zip(countries, dataset):
    #    winnerNeuron = kohonen.test(data)
    #    print(f"{country} - {winnerNeuron}")
    Utils.saveKohonenOutput("output.csv", kohonen, countries, dataset)
# -- END KOHONEN ------------------------------------------------------------

# -- OJA --------------------------------------------------------------------
def solveOja(options):
    dataset = Utils.getEuropeDataset(onlyData=True)

    oja = Oja(
        data=dataset,
        maxEpochs=options["maxEpochs"],
        initialLearningRate=options["initialLearningRate"],
        changeLearningRate=options["changeLearningRate"]
    )
    pc1 = oja.run()

    print(pc1)
# -- END OJA ----------------------------------------------------------------

# -- HOPFIELD ---------------------------------------------------------------
def solveHopfield(options):
    letters = Utils.getLettersDataset()
    patterns = []
    for l in options["letters"]:
        patterns.append(letters[ord(l) - ord("A")])
    
    hopfield = Hopfield(
        patterns=patterns,
        maxEpochs=options["maxEpochs"]
    )
    
    noisyLetter = Utils.saltAndPepper(patterns[0], options["noiseLevel"])
    patternsPerEpoch = hopfield.test(noisyLetter)
    Plotter.drawLetter((noisyLetter, patternsPerEpoch[-1]))
# -- END HOPFIELD -----------------------------------------------------------

if __name__ == "__main__":
    with open(sys.argv[1], 'r') as configFile:
        config = json.load(configFile)

        Constants(seed=config["seed"])

        match config["problem"]:
            case "KOHONEN":
                solveKohonen(config["options"])
            case "OJA":
                solveOja(config["options"])
            case "HOPFIELD":
                solveHopfield(config["options"])

    sys.exit(0)
