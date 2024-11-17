import sys
import json
from resources.font import font3 as font
from src.constants import Constants
from src.utils import Utils
from src.optimizer import OptimizerFunction
from src.autoencoder import Autoencoder
from src.plotter import Plotter


### CONVENTIONAL AUTOENCODER ################################################################################
def solveConventionalAutoencoder(config):
    letters = Utils.parseFont(font)
    inputs = Utils.generateInputs(letters[:4])
    ae = Autoencoder(
        config["encoderArchitecture"],
        OptimizerFunction.getFunction(config["hyperparameters"]["optimizer"]["type"]),
        config["hyperparameters"]["optimizer"]["options"],
        inputs
    )
    ae.train()
    outputs = []
    for input in inputs:
        output = Utils.postprocessOutput(ae.test(input))
        outputs.append(output)
    
    Plotter.drawLetters([letters[1], outputs[1], letters[2], outputs[2], letters[3], outputs[3]])
#############################################################################################################

### DENOISING AUTOENCODER ###################################################################################
def solveDenoisingAutoencoder(config):
    pass
#############################################################################################################

### VARIATIONAL AUTOENCODER #################################################################################
def solveVariationalAutoencoder(config):
    pass
#############################################################################################################


if __name__ == "__main__":
    with open(sys.argv[1], 'r') as configFile:
        config = json.load(configFile)

        batchSize = 1
        if config["hyperparameters"]["updater"]["type"].lower() == "batch":
            batchSize = -1
        elif config["hyperparameters"]["updater"]["type"].lower() == "mini-batch":
            batchSize = config["hyperparameters"]["updater"]["options"]["batchSize"]

        constants = Constants(
            epsilon=config["hyperparameters"]["epsilon"],
            maxEpochs=config["hyperparameters"]["maxEpochs"],
            batchSize=batchSize,
            seed=config["seed"],
        )

        match config["problem"]:
            case "CONVENTIONAL":
                solveConventionalAutoencoder(config)
            case "DENOISING":
                solveDenoisingAutoencoder(config)
            case "VARIATIONAL":
                solveVariationalAutoencoder(config)

    sys.exit(0)
