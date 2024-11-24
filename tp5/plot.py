import pickle
from resources.font import font3 as font
from src.utils import Utils
from src.plotter import Plotter


if __name__ == "__main__":
    # Plotter.drawLetters(Utils.parseFont(font))
    
    # Plotter.mseVsEpoch("mseVsEpoch_cae_04_24letters.csv")
    
    # Plotter.latentSpace("latentSpaceOutput_cae_04_24letters.csv")
    
    # with open("lettersAE_cae_04_24letters.pickle", 'rb') as file:
    #     ae = pickle.load(file)
    #     Plotter.newElements(
    #         ae,
    #         8,
    #         [-1, 1],
    #         [-1, 1],
    #         (7, 5)
    #     )
    
    # Plotter.drawIcons(Utils.parseIcons("icons", 8))
    # Plotter.latentSpace("latentSpaceOutput_vae.csv")
    with open("iconsVAE.pickle", 'rb') as file:
        ae = pickle.load(file)
        Plotter.newElements(
            ae,
            8,
            [-3, 3],
            [-3, 3],
            (20, 20)
        )
