import pickle
import sys
from src.constants import *
from src.printer import Printer


if __name__ == "__main__":
    with open(sys.argv[1], 'rb') as file:
        ae = pickle.load(file)

    # Printer.mseVsEpoch("mseVsEpoch_cae_01.csv", ae)
    # Printer.latentSpaceOutput("latentSpaceOutput_cae_04_24letters_batch.csv", ae, [chr(ord('a') + i) for i in range(24)])
    Printer.latentSpaceOutput("latentSpaceOutput_vae.csv", ae, ["smile", "surprise", "sunglasses", "sexy", "angry", "happy", "emotionless", "dead"])
