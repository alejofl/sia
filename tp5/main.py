import sys
from resources.font import font3 as font
from src.utils import Utils
from src.plotter import Plotter


if __name__ == "__main__":
    letters = Utils.parseFont(font)
    Plotter.drawLetters(letters)
    print(letters[-1].reshape(7, 5))

    sys.exit(0)
