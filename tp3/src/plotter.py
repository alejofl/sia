import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation


class Plotter:
    COLORS = ["#003049", "#d62828", "#f77f00", "#fcbf49", "#eae2b7", "#007F83", "#BC3C28", "#FFD56B", "#006A4E"]
    COLOR_VARIANTS = ["#335a7f", "#f05c5c", "#ffaa4d", "#ffe08b", "#f5ecd0", "#33b2b5", "#e06650", "#ffe69b", "#33997b"]
    
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

    @staticmethod
    def graphEx1(dataset, weightsPerEpoch):
        fig, ax = plt.subplots()

        ax.set_xticks([-1, 1])
        ax.set_yticks([-1, 1])

        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)

        ax.spines['left'].set_position('center')
        ax.spines['bottom'].set_position('center')

        ax.spines['right'].set_color('none')
        ax.spines['top'].set_color('none')

        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')

        scatter = ax.scatter(dataset["x1"], dataset["x2"])
        line, = ax.plot([], [], lw=2, color='black')

        def init():
            line.set_data([], [])
            return line,

        def update(epoch):
            ax.set_title(f'Separation Hyperplane (epoch {epoch})')
            x_values = np.array([-1.5, 1.5])
            m = -weightsPerEpoch[epoch][1] / weightsPerEpoch[epoch][2]
            b = -weightsPerEpoch[epoch][0] / weightsPerEpoch[epoch][2]
            y_values = m * x_values + b
            line.set_data(x_values, y_values)
            side = dataset["x2"] - (m * dataset["x1"] + b)
            colors = np.where(side >= 0, 'blue', 'red')
            scatter.set_color(colors)
            return line,

        ani = FuncAnimation(fig, update, frames=list(range(len(weightsPerEpoch))) + [len(weightsPerEpoch)-1]*20, init_func=init, blit=False)

        ani.save('perceptron_animation.gif', writer='pillow')
        
    @staticmethod
    def ex2_betas(filename):
        df = pd.read_csv(filename)
        
        fig, ax = plt.subplots()

        ax.plot(df["beta"], df["mse"], label="MSE", color=Plotter.COLORS[1])
        ax.scatter(df["beta"], df["mse"], color=Plotter.COLORS[1])
        ax.set_xscale("log")
        
        plt.xlabel("Beta")
        plt.ylabel("Mean Squared Error")
        plt.legend()
        plt.show()
        
    @staticmethod
    def ex2_mseVsEpoch_allFunctions(linearFilename, logisticFilename, tanhFilename):
        linear = pd.read_csv(linearFilename)
        logistic = pd.read_csv(logisticFilename)
        tanh = pd.read_csv(tanhFilename)

        fig, ax = plt.subplots()
        ax.set_prop_cycle("color", Plotter.COLORS)

        ax.plot(linear["epoch"], linear["trainingMSE"], label="Linear")
        ax.plot(logistic["epoch"], logistic["trainingMSE"], label="Logistic($\\beta=0.4$)")
        ax.plot(tanh["epoch"], tanh["trainingMSE"], label="Hiperbolic Tangent($\\beta=0.1$)")
        ax.set_yscale("log")
        
        plt.xlabel("Epoch")
        plt.ylabel("Training Mean Squared Error")
        plt.legend()
        plt.show()
        
    @staticmethod
    def ex2_mseVsEpoch_allSplittings(arbitraryFilename, shuffleSplitFilename, kFoldFilename):
        arbitrary = pd.read_csv(arbitraryFilename)
        shuffleSplit = pd.read_csv(shuffleSplitFilename)
        kFold = pd.read_csv(kFoldFilename)

        fig, ax = plt.subplots()
        ax.set_prop_cycle("color", Plotter.COLORS)

        ax.plot(arbitrary["epoch"], arbitrary["trainingMSE"], label="Training Set - 80% Training | 20% Testing", color=Plotter.COLORS[1])
        ax.plot(arbitrary["epoch"], arbitrary["testingMSE"], label="Testing Set - 80% Training | 20% Testing", color=Plotter.COLOR_VARIANTS[1], linestyle="--")
        
        ax.plot(shuffleSplit["epoch"], shuffleSplit["trainingMSE"], label="Training Set - Shuffle Split", color=Plotter.COLORS[0])
        ax.fill_between(shuffleSplit["epoch"], shuffleSplit["trainingMSE"] - shuffleSplit["stdTrainingMSE"], shuffleSplit["trainingMSE"] + shuffleSplit["stdTrainingMSE"], color=Plotter.COLORS[0], alpha=0.3)
        ax.plot(shuffleSplit["epoch"], shuffleSplit["testingMSE"], label="Testing Set - Shuffle Split", color=Plotter.COLOR_VARIANTS[0], linestyle="--")
        ax.fill_between(shuffleSplit["epoch"], shuffleSplit["testingMSE"] - shuffleSplit["stdTestingMSE"], shuffleSplit["testingMSE"] + shuffleSplit["stdTestingMSE"], color=Plotter.COLOR_VARIANTS[0], alpha=0.2)
        
        ax.plot(kFold["epoch"], kFold["trainingMSE"], label="Training Set - K Fold", color=Plotter.COLORS[3])
        ax.fill_between(kFold["epoch"], kFold["trainingMSE"] - kFold["stdTrainingMSE"], kFold["trainingMSE"] + kFold["stdTrainingMSE"], color=Plotter.COLORS[3], alpha=0.2)
        ax.plot(kFold["epoch"], kFold["testingMSE"], label="Testing Set - K Fold", color=Plotter.COLOR_VARIANTS[3], linestyle="--")
        ax.fill_between(kFold["epoch"], kFold["testingMSE"] - kFold["stdTestingMSE"], kFold["testingMSE"] + kFold["stdTestingMSE"], color=Plotter.COLOR_VARIANTS[3], alpha=0.2)
        
        ax.set_ylim(-2, 50)
        
        plt.xlabel("Epoch")
        plt.ylabel("Mean Squared Error")
        plt.legend()
        plt.show()
        
    @staticmethod
    def ex3_accuracyVsEpoch_allOptimizers(gradientFilename, momentumFilename, adamFilename):
        gradient = pd.read_csv(gradientFilename)
        momentum = pd.read_csv(momentumFilename)
        adam = pd.read_csv(adamFilename)

        fig, ax = plt.subplots()
        ax.set_prop_cycle("color", Plotter.COLORS)

        ax.plot(gradient["epoch"], gradient["trainingAccuracy"], label="Training Set - Gradient Descent", color=Plotter.COLORS[0])
        ax.plot(gradient["epoch"], gradient["testingAccuracy"], label="Testing Set - Gradient Descent", color=Plotter.COLOR_VARIANTS[0], linestyle="--")
        
        ax.plot(momentum["epoch"], momentum["trainingAccuracy"], label="Training Set - Momentum", color=Plotter.COLORS[1])
        ax.plot(momentum["epoch"], momentum["testingAccuracy"], label="Testing Set - Momentum", color=Plotter.COLOR_VARIANTS[1], linestyle="--")
        
        ax.plot(adam["epoch"], adam["trainingAccuracy"], label="Training Set - Adam", color=Plotter.COLORS[3])
        ax.plot(adam["epoch"], adam["testingAccuracy"], label="Testing Set - Adam", color=Plotter.COLOR_VARIANTS[3], linestyle="--")
        
        ax.set_xscale("log")
        
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.show()

    @staticmethod
    def ex3_accuracyVsEpoch_allUpdaters(onlineFilename, miniBatchFilename, batchFilename):
        online = pd.read_csv(onlineFilename)
        miniBatch = pd.read_csv(miniBatchFilename)
        batch = pd.read_csv(batchFilename)

        fig, ax = plt.subplots()
        ax.set_prop_cycle("color", Plotter.COLORS)

        ax.plot(online["epoch"], online["trainingAccuracy"], label="Training Set - Online", color=Plotter.COLORS[0])
        ax.plot(online["epoch"], online["testingAccuracy"], label="Testing Set - Online", color=Plotter.COLOR_VARIANTS[0], linestyle="--")
        
        ax.plot(miniBatch["epoch"], miniBatch["trainingAccuracy"], label="Training Set - Mini-Batch", color=Plotter.COLORS[1])
        ax.plot(miniBatch["epoch"], miniBatch["testingAccuracy"], label="Testing Set - Mini-Batch", color=Plotter.COLOR_VARIANTS[1], linestyle="--")
        
        ax.plot(batch["epoch"], batch["trainingAccuracy"], label="Training Set - Batch", color=Plotter.COLORS[3])
        ax.plot(batch["epoch"], batch["testingAccuracy"], label="Testing Set - Batch", color=Plotter.COLOR_VARIANTS[3], linestyle="--")
        
        ax.set_xscale("log")
        
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.show()
