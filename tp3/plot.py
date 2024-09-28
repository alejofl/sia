import sys
import os
from src.plotter import Plotter


if __name__ == "__main__":
    Plotter.mseVsEpoch(os.path.join(os.path.dirname(sys.argv[0]), "results", "setMSE.csv"))

    sys.exit(0)
