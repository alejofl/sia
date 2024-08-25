import json
import csv
import pandas as pd
import glob
import matplotlib.pyplot as plt
import scipy.stats as ss 

def plot_comparison(level, heur, variable, algs, title, xLabel, yLabel, ):
    csvFiles = []
    for alg in algs:
        if heur != "NO": 
            file = file = glob.glob(f'results/*/*_{alg}_{heur}_{level}_results.csv')[0]
        else:
            file = glob.glob(f'results/*/*_{alg}_{level}_results.csv')[0]
        csvFiles.append(file)

    print(csvFiles)
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
        plt.bar(resultDf['algorithm'], resultDf[variable], color=["#003049", "#d62828"], yerr=errors)
    else:
        plt.bar(resultDf['algorithm'], resultDf[variable], color=["#003049", "#d62828"])
    

    plt.title(title)
    plt.xlabel(xLabel)
    plt.ylabel(yLabel)

    plt.show()

if __name__ == "__main__":
    print("main is running!")
    level = "soko06"
    plot_comparison(level, "MANHATTAN", "meanExecutionTime", ["GREEDY", "ASTAR"], f"Greedy vs A* mean execution time - {level}", "Algorithm", "Mean execution time (s)")