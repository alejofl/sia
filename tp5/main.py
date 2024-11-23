import sys
import json
import pickle
from resources.font import font3 as font
from src.constants import Constants, LETTERS_AE_PICKLE_FILENAME, LETTERS_DAE_PICKLE_FILENAME, LETTERS_VAE_PICKLE_FILENAME
from src.utils import Utils
from src.optimizer import OptimizerFunction
from src.autoencoder import Autoencoder
from src.vae import VAE
from src.plotter import Plotter


### CONVENTIONAL AUTOENCODER ################################################################################
def solveConventionalAutoencoder(config, loadPickle=False, dumpPickle=False):
    letters = Utils.parseFont(font)
    inputs = Utils.generateInputs(letters)

    if loadPickle:
        with open(LETTERS_AE_PICKLE_FILENAME, 'rb') as file:
            ae = pickle.load(file)
    else:
        ae = Autoencoder(
            config["encoderArchitecture"],
            OptimizerFunction.getFunction(config["hyperparameters"]["optimizer"]["type"]),
            config["hyperparameters"]["optimizer"]["options"],
            inputs
        )
        ae.train()

    if dumpPickle:
        with open(LETTERS_AE_PICKLE_FILENAME, 'wb') as file:
            pickle.dump(ae, file)

    outputs = []
    for input in inputs:
        output = Utils.postprocessOutput(ae.test(input))
        outputs.append(output)
    
    Plotter.drawLetters(outputs)
#############################################################################################################

### DENOISING AUTOENCODER ###################################################################################
def solveDenoisingAutoencoder(config, loadPickle=False, dumpPickle=False):
    letters = Utils.parseFont(font)
    inputs = Utils.generateInputs(letters)

    if loadPickle:
        with open(LETTERS_DAE_PICKLE_FILENAME, 'rb') as file:
            ae = pickle.load(file)
    else:
        ae = Autoencoder(
            config["encoderArchitecture"],
            OptimizerFunction.getFunction(config["hyperparameters"]["optimizer"]["type"]),
            config["hyperparameters"]["optimizer"]["options"],
            inputs
        )
        ae.train()

    if dumpPickle:
        with open(LETTERS_DAE_PICKLE_FILENAME, 'wb') as file:
            pickle.dump(ae, file)

    toDraw = []
    for letter in letters:
        input = None
        if config["problemOptions"]["noiseType"].lower() == "gaussian":
            input = Utils.gaussianFilter(letter, config["problemOptions"]["noiseLevel"])
        else:
            input = Utils.saltAndPepperFilter(letter, config["problemOptions"]["noiseThreshold"])
        output = Utils.postprocessOutput(ae.test(Utils.generateInput(input)))
        toDraw.append(letter)
        toDraw.append(input)
        toDraw.append(output)
    
    Plotter.drawLetters(toDraw)
#############################################################################################################

### VARIATIONAL AUTOENCODER #################################################################################
def solveVariationalAutoencoder(config, loadPickle=False, dumpPickle=False):
    icons = Utils.parseIcons("icons", 8)
    inputs = Utils.generateInputs(icons)

    if loadPickle:
        with open(LETTERS_VAE_PICKLE_FILENAME, 'rb') as file:
            vae = pickle.load(file)
    else:
        vae = VAE(
            config["encoderArchitecture"],
            OptimizerFunction.getFunction(config["hyperparameters"]["optimizer"]["type"]),
            config["hyperparameters"]["optimizer"]["options"],
            inputs
        )
        vae.train()

    if dumpPickle:
        with open(LETTERS_VAE_PICKLE_FILENAME, 'wb') as file:
            pickle.dump(vae, file)

    Plotter.drawIcons(icons)
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
        
        loadPickle = False
        try:
            loadPickle = Utils.parseBoolean(sys.argv[2])
        except IndexError:
            pass
        
        dumpPickle = False
        try:
            dumpPickle = Utils.parseBoolean(sys.argv[3])
        except IndexError:
            pass

        match config["problem"].lower():
            case "conventional":
                solveConventionalAutoencoder(config, loadPickle, dumpPickle)
            case "denoising":
                solveDenoisingAutoencoder(config, loadPickle, dumpPickle)
            case "variational":
                solveVariationalAutoencoder(config, loadPickle, dumpPickle)

    sys.exit(0)
