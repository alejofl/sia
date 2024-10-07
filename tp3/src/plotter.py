import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation


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
            ax.set_title(f'Separation Hyperplane (epoch {epoch+1})')
            x_values = np.array([-1.5, 1.5])
            m = -weightsPerEpoch[epoch][1] / weightsPerEpoch[epoch][2]
            b = -weightsPerEpoch[epoch][0] / weightsPerEpoch[epoch][2]
            y_values = m * x_values + b
            line.set_data(x_values, y_values)
            side = dataset["x2"] - (m * dataset["x1"] + b)
            colors = np.where(side >= 0, 'blue', 'red')
            scatter.set_color(colors)
            return line,

        ani = FuncAnimation(fig, update, frames=len(weightsPerEpoch), init_func=init, blit=True)

        ani.save('perceptron_animation.gif', writer='imagemagick')
        
