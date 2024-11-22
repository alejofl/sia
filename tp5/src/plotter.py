import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class Plotter:
    COLORS = ["#003049", "#d62828", "#f77f00", "#fcbf49", "#eae2b7", "#007F83", "#BC3C28", "#FFD56B", "#006A4E"]
    
    @staticmethod
    def drawLetters(letters):
        rows = len(letters) // 7 + (1 if len(letters) % 7 != 0 else 0)
        cols = 7 if len(letters) > 7 else len(letters)
        fig, axs = plt.subplots(rows, cols)

        axs = axs.flatten()
        for i, (ax, letter) in enumerate(zip(axs, letters)):
            letter = np.array(letter).reshape(7, 5)
            ax.imshow(letter, cmap="Greys", interpolation="nearest")

        for i, ax in enumerate(axs):
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
            if i > len(letters) - 1:
                fig.delaxes(ax)
        plt.show()

    @staticmethod
    def mseVsEpoch(filename):
        df = pd.read_csv(filename)

        fig, ax = plt.subplots()
        ax.set_prop_cycle("color", Plotter.COLORS)

        ax.plot(df["epoch"], df["mse"])
        
        plt.yscale("log")
        plt.xlabel("Epoch")
        plt.ylabel("Mean Squared Error")
        plt.legend()
        plt.show()
