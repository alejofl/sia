import sys
from src.plotter import Plotter
from src.constants import Constants

if __name__ == "__main__":
    Constants()
    Plotter.kohonenHeatmap("output.csv", 10)
    sys.exit(0)
