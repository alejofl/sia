import sys
from src.plotter import Plotter
from src.constants import Constants

if __name__ == "__main__":
    Constants()
    # Plotter.kohonenHeatmap("output.csv", 10)
    Plotter.kohonenVariablesComparison("output.csv", 3, ["GDP", "Unemployment"])
    Plotter.kohonenVariablesComparison("output.csv", 3, ["GDP", "Inflation"])
    Plotter.kohonenVariablesComparison("output.csv", 3, ["GDP", "Life.expect"])
    Plotter.kohonenVariablesComparison("output.csv", 3, ["Life.expect", "Pop.growth"])
    Plotter.kohonenVariablesComparison("output.csv", 3, ["Life.expect", "Unemployment"])
    # Plotter.drawLetterPerEpoch("f.csv")
    sys.exit(0)
