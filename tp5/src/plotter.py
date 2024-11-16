import numpy as np
import matplotlib.pyplot as plt


class Plotter:
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
