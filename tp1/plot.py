import json
import csv
import pandas as pd
import glob
import matplotlib.pyplot as plt
import scipy.stats as ss 

COLORS = ["#003049", "#d62828", "#f77f00", "#fcbf49", "#eae2b7", "#007F83", "#BC3C28", "#FFD56B"]
ALGORITHMS = [("DFS", None),
              ("BFS", None),
              ("GREEDY", "MANHATTAN"), ("GREEDY", "EUCLIDEAN"), ("GREEDY", "BOUNDING_BOX"),
              ("ASTAR", "MANHATTAN"), ("ASTAR", "EUCLIDEAN"), ("ASTAR", "BOUNDING_BOX")]


def plot_comparison(level, heur, variable, algs, title, xLabel, yLabel):
    csvFiles = []
    for alg in algs:
        if heur != "NO": 
            file = glob.glob(f'results/*/*_{alg}_{heur}_{level}_results.csv')[0]
        else:
            file = glob.glob(f'results/*/*_{alg}_{level}_results.csv')[0]
        csvFiles.append(file)

    resultDf = pd.DataFrame()
    errors = []
    for file in csvFiles:
        df = pd.read_csv(file)
        alg_name = file.split("_")[1]
        cost = df[variable][0] if variable != "meanExecutionTime" else df["executionTime"].mean()
        if variable=="meanExecutionTime":
            errors.append(ss.sem(df["executionTime"]))
        newRow = {'algorithm': alg_name, variable: cost}
        resultDf = pd.concat([resultDf, pd.DataFrame([newRow])], ignore_index=True)
    plt.figure(figsize=(8, 6))
    if variable=="meanExecutionTime":
        plt.bar(resultDf['algorithm'], resultDf[variable], color=COLORS, yerr=errors)
    else:
        plt.bar(resultDf['algorithm'], resultDf[variable], color=COLORS)

    plt.title(title)
    plt.xlabel(xLabel)
    plt.ylabel(yLabel)

    plt.show()


def plot_all_algorithms(level, variable, title, xLabel, yLabel, algorithms = ALGORITHMS):
    csvs = []
    for algorithm, heuristic in algorithms:
        if heuristic:
            csvs.append(glob.glob(f'results/*/*_{algorithm}_{heuristic}_{level}_results.csv')[0])
        else:
            csvs.append(glob.glob(f'results/*/*_{algorithm}_{level}_results.csv')[0])
    
    resultDf = pd.DataFrame()
    errors = []
    for file, (algorithm, heuristic) in zip(csvs, algorithms):
        df = pd.read_csv(file)
        alg_name = f"{algorithm} - {heuristic}" if heuristic else algorithm
        cost = df[variable][0] if variable != "meanExecutionTime" else df["executionTime"].mean()
        if variable=="meanExecutionTime":
            errors.append(ss.sem(df["executionTime"]))
        newRow = {'algorithm': alg_name, variable: cost}
        resultDf = pd.concat([resultDf, pd.DataFrame([newRow])], ignore_index=True)
    
    fig, ax = plt.subplots()
    
    if variable=="meanExecutionTime":
        ax.bar(resultDf['algorithm'], resultDf[variable], color=COLORS, yerr=errors)
    else:
        ax.bar(resultDf['algorithm'], resultDf[variable], color=COLORS)
    
    plt.title(title)
    plt.xlabel(xLabel)
    plt.ylabel(yLabel)

    plt.show()

if __name__ == "__main__":
    print("plot is running!")
    level = "soko06"
    # plot_comparison(level, "MANHATTAN", "meanExecutionTime", ["GREEDY", "ASTAR"], f"Greedy vs A* mean execution time - {level}", "Algorithm", "Mean execution time (s)")
    plot_all_algorithms(level, "cost", f"A* Heuristic Cost - {level}", "Algorithm", "Cost (steps)", [("ASTAR", "MANHATTAN"), ("ASTAR", "EUCLIDEAN"), ("ASTAR", "BOUNDING_BOX")])