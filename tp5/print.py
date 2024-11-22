import pickle
import sys
from src.constants import *
from src.printer import Printer


if __name__ == "__main__":
    with open(sys.argv[1], 'rb') as file:
        ae = pickle.load(file)

    Printer.mseVsEpoch("mseVsEpoch_cae_01.csv", ae)
