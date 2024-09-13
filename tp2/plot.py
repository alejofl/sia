import pandas as pd
import glob
import matplotlib.pyplot as plt
import scipy.stats as ss 

COLORS = ["#003049", "#d62828", "#f77f00", "#fcbf49", "#eae2b7", "#007F83", "#BC3C28", "#FFD56B"]

def comparison(results, variable, title, xLabel, yLabel):

    resultDf = pd.DataFrame()
    errors = []

    for crossoverTuple in results:
        df = pd.read_csv(crossoverTuple[0])
        variableDf = df[variable]
        newRow = {'method': crossoverTuple[1], variable: variableDf.mean()}
        errors.append(ss.sem(variableDf))
        resultDf = pd.concat([resultDf, pd.DataFrame([newRow])], ignore_index=True)

    plt.figure(figsize=(8, 6))
    
    min_value = resultDf[variable].min()
    max_value = resultDf[variable].max()
    
    range_value = max_value - min_value
    lower_limit = min_value - 0.1 * range_value

    plt.bar(resultDf['method'], resultDf[variable], color=COLORS, yerr=errors)
    plt.ylim(lower_limit, max_value + 0.05 * range_value)
    # plt.axhline(max_value, color='black', linestyle='--', label=f'Max: {max_value:.2f}')
    # for index, value in enumerate(resultDf[variable]): 
    #     plt.text(index, value, f'{value:.6f}', ha='center', va='bottom')

    if variable=='fitness':
            plt.text(resultDf[variable].idxmax(), max_value, f'{max_value:.6f}', ha='center', va='bottom', fontweight='bold')



    plt.title(title)
    plt.xlabel(xLabel)
    plt.ylabel(yLabel)

    plt.show()

fitnessLabel = "Fitness (%)"
executionTimeLabel = "Execution time (s)"
mutationCount = "Mutations"



if __name__ == "__main__":
    print("Plotter is running")
    # This function receives a list of tuples [method results filename, method name to show)] and a tuple [variable for comparison in csv, variable name to show]

    #Diapo 11
    # comparison([['results/crossover/ONE_POINT-results.csv', 'One Point'], ['results/crossover/TWO_POINTS-results.csv', 'Two Points'], ['results/crossover/ANULAR-results.csv', 'Anular'], ['results/crossover/UNIFORM_p06-results.csv', 'Uniform (p=0.6)']], 'fitness', "Crossover: Fitness", "Crossover method", 'Fitness (%)')
    # comparison([['results/crossover/ONE_POINT-results.csv', 'One Point'], ['results/crossover/TWO_POINTS-results.csv', 'Two Points'], ['results/crossover/ANULAR-results.csv', 'Anular'], ['results/crossover/UNIFORM_p06-results.csv', 'Uniform (p=0.6)']], 'executionTime', "Crossover: Execution Time", "Crossover method", 'Execution Time (s)')

    #Diapo 12
    # comparison([['results/crossover/UNIFORM_p01-results.csv', 'p=0.1'], ['results/crossover/UNIFORM_p02-results.csv', 'p=0.2'], ['results/crossover/UNIFORM_p03-results.csv', 'p=0.3'], ['results/crossover/UNIFORM_p04-results.csv', 'p=0.4'], ['results/crossover/UNIFORM_p05-results.csv', 'p=0.5'], ['results/crossover/UNIFORM_p06-results.csv', 'p=0.6'], ['results/crossover/UNIFORM_p07-results.csv', 'p=0.7'], ['results/crossover/UNIFORM_p08-results.csv', 'p=0.8'], ['results/crossover/UNIFORM_p09-results.csv', 'p=0.9']], 'fitness', "Uniform cross method probabilities: Fitness", "Uniform probability", 'Fitness (%)')
    # comparison([['results/crossover/UNIFORM_p01-results.csv', 'p=0.1'], ['results/crossover/UNIFORM_p02-results.csv', 'p=0.2'], ['results/crossover/UNIFORM_p03-results.csv', 'p=0.3'], ['results/crossover/UNIFORM_p04-results.csv', 'p=0.4'], ['results/crossover/UNIFORM_p05-results.csv', 'p=0.5'], ['results/crossover/UNIFORM_p06-results.csv', 'p=0.6'], ['results/crossover/UNIFORM_p07-results.csv', 'p=0.7'], ['results/crossover/UNIFORM_p08-results.csv', 'p=0.8'], ['results/crossover/UNIFORM_p09-results.csv', 'p=0.9']], 'executionTime', "Uniform cross method probabilities: Execution Time", "Uniform probability", 'Execution Time (s)')

    #Diapo 13
    # comparison([['results/crossover/ONE_POINT-results.csv', 'One Point'], ['results/crossover/TWO_POINTS-results.csv', 'Two Points'], ['results/crossover/ANULAR-results.csv', 'Anular'], ['results/crossover/UNIFORM_p03-results.csv', 'Uniform (p=0.3)']], 'fitness', "Crossover: Fitness", "Crossover method", 'Fitness (%)')
    # comparison([['results/crossover/ONE_POINT-results.csv', 'One Point'], ['results/crossover/TWO_POINTS-results.csv', 'Two Points'], ['results/crossover/ANULAR-results.csv', 'Anular'], ['results/crossover/UNIFORM_p03-results.csv', 'Uniform (p=0.3)']], 'executionTime', "Crossover: Execution Time", "Crossover method", 'Execution Time (s)')

    # comparison(
    #     [['results/mutation/gen-0,1-results.csv', 'p=0.1'], ['results/mutation/gen-0,2-results.csv', 'p=0.2'],
    #      ['results/mutation/gen-0,3-results.csv', 'p=0.3'], ['results/mutation/gen-0,4-results.csv', 'p=0.4'],
    #      ['results/mutation/gen-0,5-results.csv', 'p=0.5'], ['results/mutation/gen-0,6-results.csv', 'p=0.6'],
    #      ['results/mutation/gen-0,7-results.csv', 'p=0.7'], ['results/mutation/gen-0,8-results.csv', 'p=0.8'],
    #      ['results/mutation/gen-0,9-results.csv', 'p=0.9']], 'fitness',
    #     "Gen mutation method probabilities: Fitness", "Mutation probability", 'Fitness (%)')
    # comparison(
    #     [['results/mutation/gen-0,1-results.csv', 'p=0.1'], ['results/mutation/gen-0,2-results.csv', 'p=0.2'],
    #      ['results/mutation/gen-0,3-results.csv', 'p=0.3'], ['results/mutation/gen-0,4-results.csv', 'p=0.4'],
    #      ['results/mutation/gen-0,5-results.csv', 'p=0.5'], ['results/mutation/gen-0,6-results.csv', 'p=0.6'],
    #      ['results/mutation/gen-0,7-results.csv', 'p=0.7'], ['results/mutation/gen-0,8-results.csv', 'p=0.8'],
    #      ['results/mutation/gen-0,9-results.csv', 'p=0.9']], 'executionTime',
    #     "Gen mutation method probabilities: Execution Time", "Mutation probability", 'Execution Time (s)')
    # comparison(
    #     [['results/mutation/gen-0,1-results.csv', 'p=0.1'], ['results/mutation/gen-0,2-results.csv', 'p=0.2'],
    #      ['results/mutation/gen-0,3-results.csv', 'p=0.3'], ['results/mutation/gen-0,4-results.csv', 'p=0.4'],
    #      ['results/mutation/gen-0,5-results.csv', 'p=0.5'], ['results/mutation/gen-0,6-results.csv', 'p=0.6'],
    #      ['results/mutation/gen-0,7-results.csv', 'p=0.7'], ['results/mutation/gen-0,8-results.csv', 'p=0.8'],
    #      ['results/mutation/gen-0,9-results.csv', 'p=0.9']], 'mutationCount',
    #     "Gen mutation method probabilities: Mutation count", "Mutation probability", 'Mutations')

    # comparison(
    #     [['results/mutation/multigen-0,1-results.csv', 'p=0.1'], ['results/mutation/multigen-0,2-results.csv', 'p=0.2'],
    #      ['results/mutation/multigen-0,3-results.csv', 'p=0.3'], ['results/mutation/multigen-0,4-results.csv', 'p=0.4'],
    #      ['results/mutation/multigen-0,5-results.csv', 'p=0.5'], ['results/mutation/multigen-0,6-results.csv', 'p=0.6'],
    #      ['results/mutation/multigen-0,7-results.csv', 'p=0.7'], ['results/mutation/multigen-0,8-results.csv', 'p=0.8'],
    #      ['results/mutation/multigen-0,9-results.csv', 'p=0.9']], 'fitness',
    #     "Multigen mutation method probabilities: Fitness", "Mutation probability", 'Fitness (%)')
    # comparison(
    #     [['results/mutation/multigen-0,1-results.csv', 'p=0.1'], ['results/mutation/multigen-0,2-results.csv', 'p=0.2'],
    #      ['results/mutation/multigen-0,3-results.csv', 'p=0.3'], ['results/mutation/multigen-0,4-results.csv', 'p=0.4'],
    #      ['results/mutation/multigen-0,5-results.csv', 'p=0.5'], ['results/mutation/multigen-0,6-results.csv', 'p=0.6'],
    #      ['results/mutation/multigen-0,7-results.csv', 'p=0.7'], ['results/mutation/multigen-0,8-results.csv', 'p=0.8'],
    #      ['results/mutation/multigen-0,9-results.csv', 'p=0.9']], 'executionTime',
    #     "Multigen mutation method probabilities: Execution Time", "Mutation probability", 'Execution Time (s)')
    
    # comparison(
    #     [['results/mutation/multigen-0,1-results.csv', 'p=0.1'], ['results/mutation/multigen-0,2-results.csv', 'p=0.2'],
    #      ['results/mutation/multigen-0,3-results.csv', 'p=0.3'], ['results/mutation/multigen-0,4-results.csv', 'p=0.4'],
    #      ['results/mutation/multigen-0,5-results.csv', 'p=0.5'], ['results/mutation/multigen-0,6-results.csv', 'p=0.6'],
    #      ['results/mutation/multigen-0,7-results.csv', 'p=0.7'], ['results/mutation/multigen-0,8-results.csv', 'p=0.8'],
    #      ['results/mutation/multigen-0,9-results.csv', 'p=0.9']], 'mutationCount',
    #     "Multigen mutation method probabilities: Mutation count", "Mutation probability", 'Mutations')
    
    # comparison(
    #     [['results/mutation/gen-0,7-results.csv', 'Gen (p=0.7)'], ['results/mutation/gen-0,8-results.csv', 'Gen (p=0.8)'], ['results/mutation/multigen-0,1-results.csv', 'Multigen (p=0.1)'], ['results/mutation/multigen-0,2-results.csv', 'Multigen (p=0.2)']], 'fitness',
    #     "Gen vs Multigen mutation: Best fitness", "Mutation method", fitnessLabel)
    
    # comparison(
    #     [['results/mutation/gen-0,7-results.csv', 'Gen (p=0.7)'], ['results/mutation/gen-0,8-results.csv', 'Gen (p=0.8)'], ['results/mutation/multigen-0,1-results.csv', 'Multigen (p=0.1)'], ['results/mutation/multigen-0,2-results.csv', 'Multigen (p=0.2)']], 'executionTime',
    #     "Gen vs Multigen mutation: Execution time", "Mutation method", executionTimeLabel)

    comparison([['results/selection/detTournament-2,2%-results.csv', 'm=22'],['results/selection/detTournament-2,4%-results.csv', 'm=24'],['results/selection/detTournament-2,5%-results.csv', 'm=25'],['results/selection/detTournament-2,6%-results.csv', 'm=26'],['results/selection/detTournament-2,7%-results.csv', 'm=27'],['results/selection/detTournament-5%-results.csv', 'm=50'], ['results/selection/detTournament-7,5%-results.csv', 'm=75'],['results/selection/detTournament-10%-results.csv', 'm=100'],['results/selection/detTournament-25%-results.csv', 'm=250'], ['results/selection/detTournament-50%-results.csv', 'm=500'],
          ['results/selection/detTournament-75%-results.csv', 'm=750']], 'fitness', "Deterministic Tournament Selection: Fitness", "Selection method", 'Fitness (%)')