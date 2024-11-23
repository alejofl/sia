import pickle
from src.plotter import Plotter


if __name__ == "__main__":
    # Plotter.mseVsEpoch("mseVsEpoch_cae_04_24letters.csv")
    
    # Plotter.latentSpace("latentSpaceOutput_cae_04_24letters.csv")
    
    with open("lettersAE_cae_04_24letters.pickle", 'rb') as file:
        ae = pickle.load(file)
        Plotter.newElements(
            ae,
            8,
            [-1, 1],
            [-1, 1],
            (7, 5)
        )
