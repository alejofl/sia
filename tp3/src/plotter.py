import pandas as pd
import matplotlib.pyplot as plt


class Plotter:
    COLORS = ["#003049", "#d62828", "#f77f00", "#fcbf49", "#eae2b7", "#007F83", "#BC3C28", "#FFD56B"]
    
    @staticmethod
    def mseVsEpoch(filename):
        df = pd.read_csv(filename)

        fig, ax = plt.subplots()
        ax.set_prop_cycle("color", Plotter.COLORS)

        ax.plot(df["epoch"], df["trainingMSE"], label="Training Set")
        ax.plot(df["epoch"], df["testingMSE"], label="Testing Set")
        
        plt.xlabel("Epoch")
        plt.ylabel("Mean Squared Error")
        plt.legend()
        plt.show()
