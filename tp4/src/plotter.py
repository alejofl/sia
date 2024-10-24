import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from .constants import Constants


class Plotter:
    @staticmethod
    def kohonenHeatmap(filename, k):
        filepath = os.path.join(os.path.dirname(sys.argv[0]), "results", "kohonen", filename)
        df = pd.read_csv(filepath)
        values = np.zeros((k, k))
        labels = [[""] * k for _ in range(k)]
        for _, row in df.iterrows():
            i, j = int(row['row']), int(row['col'])
            values[i][j] += 1
            labels[i][j] += Constants.getInstance().countryAbbreviations.get(row["Country"]) + '\n'
        fig, ax = plt.subplots()
        im = ax.imshow(values, cmap="magma")
        ax.figure.colorbar(im, ax=ax)
        ax.set_xticks(np.arange(k))
        ax.set_yticks(np.arange(k))

        for i in range(k):
            for j in range(k):
                text_color = "white" if values[i][j] < 1.5 else "black"
                ax.text(j, i, labels[i][j][:-1], ha="center", va="center", color=text_color)

        fig.tight_layout()
        plt.show()

    @staticmethod
    def drawLetter(letters):
        fig, axs = plt.subplots(1, len(letters))
        for i, (ax, letter) in enumerate(zip(axs, letters)):
            letter = np.array(letter).reshape(5, 5)
            ax.imshow(letter, cmap="Greys", interpolation="nearest")
            ax.tick_params(
                axis='both',
                which='both',
                bottom=False,
                top=False,
                left=False,
                right=False,
                labelbottom=False,
                labeltop=False,
                labelleft=False,
                labelright=False
            )
            ax.set_title("Noisy Letter" if i == 0 else "Predicted Letter")
        plt.show()
