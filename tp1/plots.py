import json
import csv
import pandas as pd
import glob
import matplotlib.pyplot as plt
import numpy as np

def plot_all_algs(level, heur, variable):
    csv_files_no_heur = glob.glob(f'results/{level}/*_*_{level}_results.csv')
    csv_files_heur = glob.glob(f'results/{level}/*_*_{heur}_{level}_results.csv')
    csv_files = csv_files_no_heur + csv_files_heur

    result_df = pd.DataFrame()
    for file in csv_files:
        df = pd.read_csv(file)
        alg_name = file.split("_")[1]
        cost = df[variable][0] if variable != "meanExecutionTime" else df["executionTime"].mean()
        new_row = {'algorithm': alg_name, variable: cost}
        result_df = pd.concat([result_df, pd.DataFrame([new_row])], ignore_index=True)

    plt.figure(figsize=(8, 6))
    plt.bar(result_df['algorithm'], result_df[variable], color=['blue', 'green'])

    plt.title(variable + ' of Algorithms')
    plt.xlabel('Algorithm')
    plt.ylabel(variable)

    plt.show()


def plot_bfs_vs_dfs_cost(level):
    bfs_csv = glob.glob(f'results/{level}/*_BFS_{level}_results.csv')
    dfs_csv = glob.glob(f'results/{level}/*_DFS_{level}_results.csv')
    csv_files = bfs_csv + dfs_csv

    result_df = pd.DataFrame()
    for file in csv_files:
        print("reading: ",file)
        df = pd.read_csv(file)
        alg_name = file.split("_")[1]
        cost = df["cost"][0]
        result_df = pd.concat([result_df, pd.DataFrame([{'algorithm': alg_name, 'cost': cost}])],
            ignore_index=True)

    plt.figure(figsize=(8, 6))
    plt.bar(result_df['algorithm'], result_df['cost'], color=['blue', 'green'])

    plt.title('Cost of Algorithms')
    plt.xlabel('Algorithm')
    plt.ylabel('Amount of steps till completion')

    plt.show()


def plot_bfs_vs_dfs_costs():
    bfs_csvs = []
    dfs_csvs = []
    bfs_costs = []
    dfs_costs = []
    levels = ['soko01','soko02','soko03','soko04','soko05','soko06']
    for i in range(1,7):
        bfs_csvs += glob.glob(f'results/soko0{i}/*_BFS_soko0{i}_results.csv')
        dfs_csvs += glob.glob(f'results/soko0{i}/*_DFS_soko0{i}_results.csv')

    for file in bfs_csvs + dfs_csvs:
        df = pd.read_csv(file)
        alg_name = file.split("_")[1]
        match alg_name:
            case 'BFS':
                bfs_costs.append(df["cost"][0])
            case 'DFS':
                dfs_costs.append(df["cost"][0])

    index = np.arange(len(levels))
    width = 0.35  # ancho de las barras

    plt.figure(figsize=(8, 6))

    plt.bar(index, bfs_costs, width=width, label='BFS')
    plt.bar(index + width, dfs_costs, width=width, label='DFS')

    plt.xlabel('Algorithms')
    plt.ylabel('Amount of steps till completion')
    plt.title('Cost of Algorithms in Different Levels')
    plt.xticks(index + width / 2, levels)
    plt.legend()

    plt.show()


def plot_bfs_vs_dfs_exectime():
    bfs_csvs = []
    dfs_csvs = []
    bfs_exectime = []
    dfs_exectime = []
    levels = ['soko01','soko02','soko03','soko04','soko05','soko06']
    for i in range(1,7):
        bfs_csvs += glob.glob(f'results/soko0{i}/*_BFS_soko0{i}_results.csv')
        dfs_csvs += glob.glob(f'results/soko0{i}/*_DFS_soko0{i}_results.csv')

    for file in bfs_csvs + dfs_csvs:
        df = pd.read_csv(file)
        alg_name = file.split("_")[1]
        match alg_name:
            case 'BFS':
                print(df["executionTime"].mean())
                bfs_exectime.append(df["executionTime"].mean())
            case 'DFS':
                dfs_exectime.append(df["executionTime"].mean())

    index = np.arange(len(levels))
    width = 0.35  # ancho de las barras

    plt.figure(figsize=(8, 6))

    plt.bar(index, bfs_exectime, width=width, label='BFS')
    plt.bar(index + width, dfs_exectime, width=width, label='DFS')

    plt.xlabel('Algorithms')
    plt.ylabel('Time in seconds')
    plt.title('Execution Time of Algorithms in Different Levels')
    plt.xticks(index + width / 2, levels)
    plt.legend()

    plt.show()


if __name__ == "__main__":
    print("Running plots")
    #plot_all_algs("soko01", "MANHATTAN", "meanExecutionTime")
    #plot_bfs_vs_dfs_cost("soko02") #soko02 is dummy level
    #plot_bfs_vs_dfs_costs() # graph that shows cost per level per alg
    plot_bfs_vs_dfs_exectime()
