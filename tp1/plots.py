import json
import csv
import pandas as pd
import glob
import matplotlib.pyplot as plt

def plot_all_algs(level, heur, variable):
    csv_files_no_heur = glob.glob(f'results/*_*_{level}_results.csv')
    csv_files_heur = glob.glob(f'results/*_*_{heur}_{level}_results.csv')
    csv_files = csv_files_no_heur + csv_files_heur

    print(csv_files)
    result_df = pd.DataFrame()
    for file in csv_files:
        df = pd.read_csv(file)
        alg_name = file.split("_")[1]
        cost = df[variable][0] if variable != "meanExecutionTime" else df["executionTime"].mean() 
        new_row = {'algorithm': alg_name, variable: cost}
        result_df = pd.concat([result_df, pd.DataFrame([new_row])], ignore_index=True)
    print(result_df)
    plt.figure(figsize=(8, 6))
    plt.bar(result_df['algorithm'], result_df[variable], color=['blue', 'green'])

    plt.title(variable+' of Algorithms')
    plt.xlabel('Algorithm')
    plt.ylabel(variable)

    plt.show()

if __name__ == "__main__":
    print("main is running!")
    plot_all_algs("soko01", "MANHATTAN", "meanExecutionTime")

