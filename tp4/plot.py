import sys
from src.plotter import Plotter
from src.constants import Constants

if __name__ == "__main__":
    Constants()
    # Plotter.kohonenHeatmap("output.csv", 10)
    # Plotter.kohonenVariablesComparison("output.csv", 3, ["GDP", "Unemployment"])
    # Plotter.kohonenVariablesComparison("output.csv", 3, ["GDP", "Inflation"])
    # Plotter.kohonenVariablesComparison("output.csv", 3, ["GDP", "Life.expect"])
    # Plotter.kohonenVariablesComparison("output.csv", 3, ["Life.expect", "Pop.growth"])
    # Plotter.kohonenVariablesComparison("output.csv", 3, ["Life.expect", "Unemployment"])
    Plotter.pcaVsOjaLoads(
        ("Area", "GDP", "Inflation", "Life.expect", "Military", "Pop.growth", "Unemployment"),
        (-0.1248739, 0.50050586, -0.40651815, 0.48287333, -0.18811162, 0.47570355, -0.27165582),
        (0.12495734, -0.50049711, 0.40659913, -0.4828901, 0.188041, -0.475686, 0.27161368)
    )
    # Plotter.drawLetterPerEpoch("f.csv")
    sys.exit(0)
