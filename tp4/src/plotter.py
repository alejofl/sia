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
            # labels[i][j] += Constants.getInstance().countryAbbreviations.get(row["Country"]) + '\n'
            labels[i][j] += row["Country"] + '\n'
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
    def averageDistanceHeatmap(distances):
        k = len(distances) 
        fig, ax = plt.subplots()
        im = ax.imshow(distances, cmap="magma")

        ax.figure.colorbar(im, ax=ax)

        ax.set_xticks(np.arange(k))
        ax.set_yticks(np.arange(k))

        for i in range(k):
            for j in range(k):
                text_color = "white" if distances[i][j] < 5 else "black"
                ax.text(j, i, f"{distances[i][j]:.2f}", ha="center", va="center", color=text_color)

        fig.tight_layout()
        plt.show()
        
    @staticmethod
    def kohonenVariablesComparison(filename, k, variables):
        filepath = os.path.join(os.path.dirname(sys.argv[0]), "results", "kohonen", filename)
        df = pd.read_csv(filepath)
        values = np.zeros((len(variables), k, k))
        for _, row in df.iterrows():
            i, j = int(row['row']), int(row['col'])
            for idx, var in enumerate(variables):
                values[idx][i][j] += row[var]

        fig, axs = plt.subplots(1, len(variables))
        for i, (var, ax) in enumerate(zip(variables, axs)):
            im = ax.imshow(values[i], cmap="magma")
            ax.figure.colorbar(im, ax=ax)
            ax.set_xticks(np.arange(k))
            ax.set_yticks(np.arange(k))
            ax.set_title(var)

        fig.tight_layout()
        plt.show()
        
    @staticmethod
    def pcaVsOjaLoads(labels, pca, oja):
        COLORS = ("#651b80", "#dd5238")
        values = {
            "PCA": np.abs(pca),
            "OJA": np.abs(oja)
        }

        x = np.arange(len(labels))  # the label locations
        width = 0.4 # the width of the bars
        multiplier = 0

        fig, ax = plt.subplots(layout='constrained')

        for (attribute, measurement), color in zip(values.items(), COLORS):
            offset = width * multiplier
            ax.bar(x + offset, measurement, width, label=attribute, color=color, align='center')
            multiplier += 1

        ax.set_xticks(x + width / 2, labels)
        ax.legend(loc='upper left')

        plt.show()

    @staticmethod
    def drawLetter(letters, showTitle=True):
        rows = len(letters) // 7 + (1 if len(letters) % 7 != 0 else 0)
        cols = 7 if len(letters) > 7 else len(letters)
        fig, axs = plt.subplots(rows, cols)
        axs = axs.flatten()
        for i, (ax, letter) in enumerate(zip(axs, letters)):
            letter = np.array(letter).reshape(5, 5)
            ax.imshow(letter, cmap="Greys", interpolation="nearest")
            if showTitle:
                ax.set_title("Noisy Letter" if i == 0 else "Predicted Letter")
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
    def drawLetterPerEpoch(filename):
        filepath = os.path.join(os.path.dirname(sys.argv[0]), "results", "hopfield", filename)
        df = pd.read_csv(filepath, header=None)
        letters = []
        for _, row in df.iterrows():
            letters.append(row.to_numpy())
        Plotter.drawLetter(letters, showTitle=False)
        
    @staticmethod
    def energyPerEpoch(statesFilename, weightsFilename):
        statesFilepath = os.path.join(os.path.dirname(sys.argv[0]), "results", "hopfield", statesFilename)
        weightsFilepath = os.path.join(os.path.dirname(sys.argv[0]), "results", "hopfield", weightsFilename)
        df = pd.read_csv(statesFilepath, header=None)
        states = []
        for _, row in df.iterrows():
            states.append(row.to_numpy())
        weights = pd.read_csv(weightsFilepath, header=None).to_numpy()
        
        xs = []
        ys = []
        for epoch, state in enumerate(states):
            energy = 0
            for i in range(len(weights)):
                for j in range(len(weights[0])):
                    energy += weights[i][j] * state[i] * state[j]
            energy = - 0.5 * energy
            xs.append(epoch)
            ys.append(energy)
        
        fig, ax = plt.subplots()
        ax.scatter(xs, ys, color="#651b80")
        ax.plot(xs, ys, color="#651b80")
        
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Energy")
        plt.show()