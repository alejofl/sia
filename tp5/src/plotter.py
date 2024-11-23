import numpy as np
import pandas as pd
import os
import sys
import matplotlib.pyplot as plt


class Plotter:
    COLORS = ["#003049", "#d62828", "#f77f00", "#fcbf49", "#eae2b7", "#007F83", "#BC3C28", "#FFD56B", "#006A4E"]
    
    @staticmethod
    def drawLetters(letters):
        rows = len(letters) // 8 + (1 if len(letters) % 8 != 0 else 0)
        cols = 8 if len(letters) > 8 else len(letters)
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
        filepath = os.path.join(os.path.dirname(sys.argv[0]), "results", filename)
        df = pd.read_csv(filepath)

        fig, ax = plt.subplots()
        ax.set_prop_cycle("color", Plotter.COLORS)

        ax.plot(df["epoch"], df["mse"])

        plt.yscale("log")
        plt.xlabel("Epoch")
        plt.ylabel("Mean Squared Error")
        plt.show()
        
    @staticmethod
    def latentSpace(filename):
        filepath = os.path.join(os.path.dirname(sys.argv[0]), "results", filename)
        df = pd.read_csv(filepath)

        fig, ax = plt.subplots()

        for _, row in df.iterrows():
            ax.scatter(row["x"], row["y"], color=Plotter.COLORS[0])
            ax.annotate(row["letter"], (row["x"] + 0.02, row["y"] + 0.02))

        plt.xlabel("x")
        plt.ylabel("y")
        plt.show()

    @staticmethod
    def newElements(autoencoder, partitions, xRange, yRange, imageShape):
        figure = np.zeros((imageShape[0] * partitions, imageShape[1] * partitions))

        xGrid = np.linspace(*xRange, partitions)
        yGrid = np.linspace(*yRange, partitions)

        for i, y in enumerate(xGrid):
            for j, x in enumerate(yGrid):
                z = np.array([x, y])
                output = autoencoder.testWithLatentSpaceInput(z).reshape(*imageShape)
                figure[(len(xGrid) - i - 1) * imageShape[0]:(len(xGrid) - i) * imageShape[0], j * imageShape[1]:(j + 1) * imageShape[1]] = output

        fig, ax = plt.subplots()
        ax.imshow(figure, cmap="Greys", interpolation="nearest")
        ax.set_xticks(np.arange(0, partitions * imageShape[1], imageShape[1]))
        ax.set_yticks(np.arange(0, partitions * imageShape[0], imageShape[0]))
        ax.xaxis.set_major_formatter(lambda x, pos: str(np.round([np.interp(x, [0, partitions * imageShape[1]], xRange)],2)[0]))
        ax.yaxis.set_major_formatter(lambda x, pos: str(np.round([np.interp(partitions * imageShape[0] - x, [0, partitions * imageShape[0]], yRange)],2)[0]))
        plt.show()

    @staticmethod
    def drawIcons(icons):
        rows = len(icons) // 4 + (1 if len(icons) % 4 != 0 else 0)
        cols = 4 if len(icons) > 4 else len(icons)
        fig, axs = plt.subplots(rows, cols)

        axs = axs.flatten()
        for i, (ax, icon) in enumerate(zip(axs, icons)):
            icon = np.array(icon).reshape(20, 20)
            ax.imshow(icon, cmap="Greys", interpolation="nearest")

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
            if i > len(icons) - 1:
                fig.delaxes(ax)
        plt.show()
