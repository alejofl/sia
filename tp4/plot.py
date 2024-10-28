import sys
from src.plotter import Plotter
from src.constants import Constants
from src.utils import Utils

if __name__ == "__main__":
    Constants()
    # Plotter.kohonenHeatmap("output.csv", 10)
    # Plotter.kohonenVariablesComparison("output.csv", 3, ["GDP", "Unemployment"])
    # Plotter.kohonenVariablesComparison("output.csv", 3, ["GDP", "Inflation"])
    # Plotter.kohonenVariablesComparison("output.csv", 3, ["GDP", "Life.expect"])
    # Plotter.kohonenVariablesComparison("output.csv", 3, ["Life.expect", "Pop.growth"])
    # Plotter.kohonenVariablesComparison("output.csv", 3, ["Life.expect", "Unemployment"])
    
    # Plotter.pcaVsOjaLoads(
    #     ("Area", "GDP", "Inflation", "Life.expect", "Military", "Pop.growth", "Unemployment"),
    #     (-0.1248739, 0.50050586, -0.40651815, 0.48287333, -0.18811162, 0.47570355, -0.27165582),
    #     (0.12495734, -0.50049711, 0.40659913, -0.4828901, 0.188041, -0.475686, 0.27161368)
    # )
    
    # characters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    # letters = Utils.getLettersDataset()
    # patterns = []
    # for l in characters:
    #     patterns.append(letters[ord(l) - ord("A")])
    # Plotter.drawLetter(patterns, showTitle=False)
    
    # characters = "FIOVORTVAILZEOTWHMNW"
    # letters = Utils.getLettersDataset()
    # patterns = []
    # pattern = []
    # for i, l in enumerate(characters):
    #     pattern.append(letters[ord(l) - ord("A")])
    #     if i % 4 == 3:
    #         patterns.append(pattern)
    #         pattern = []
    # for p in patterns:
    #     Plotter.drawLetter(p, showTitle=False)
    
    files = ["ortv/o_0p15", "ortv/o_0p25", "ortv/o_0p35", "ortv/o_0p40", "ortv/o_0p5", "ortv/o_0p6", "fiov/o_0p05", "fiov/o_0p1", "fiov/o_0p15", "fiov/o_0p17", "fiov/o_0p19", "hmnw/h_0p01", "hmnw/h_0p05"]
    for f in files:
        print(f)
        Plotter.drawLetterPerEpoch(f + ".csv")
        Plotter.energyPerEpoch(f + ".csv", f + "_weights.csv")

    sys.exit(0)
