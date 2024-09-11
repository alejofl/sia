import pandas as pd
import glob
import matplotlib.pyplot as plt
import scipy.stats as ss 

COLORS = ["#003049", "#d62828", "#f77f00", "#fcbf49", "#eae2b7", "#007F83", "#BC3C28", "#FFD56B"]

def crossoverComparison(results, variable, title, xLabel, yLabel):

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

    plt.title(title)
    plt.xlabel(xLabel)
    plt.ylabel(yLabel)

    plt.show()


if __name__ == "__main__":
    print("Plotter is running")
    # This function receives a list of tuples [method results filename, method name to show)] and a tuple [variable for comparison in csv, variable name to show]
    # crossoverComparison([['results/crossover/ONE_POINT-results.csv', 'One Point'], ['results/crossover/TWO_POINTS-results.csv', 'Two Points'], ['results/crossover/ANULAR-results.csv', 'Anular'], ['results/crossover/UNIFORM_p03-results.csv', 'Uniform (p=0.3)']], ['fitness', 'Fitness (%)'])
    # crossoverComparison([['results/crossover/ONE_POINT-results.csv', 'One Point'], ['results/crossover/TWO_POINTS-results.csv', 'Two Points'], ['results/crossover/ANULAR-results.csv', 'Anular'], ['results/crossover/UNIFORM_p03-results.csv', 'Uniform (p=0.3)']], ['executionTime', 'Execution time (s)'])

    crossoverComparison([['results/crossover/UNIFORM_p01-results.csv', 'p=0.1'], ['results/crossover/UNIFORM_p02-results.csv', 'p=0.2'], ['results/crossover/UNIFORM_p03-results.csv', 'p=0.3'], ['results/crossover/UNIFORM_p04-results.csv', 'p=0.4'], ['results/crossover/UNIFORM_p05-results.csv', 'p=0.5'], ['results/crossover/UNIFORM_p06-results.csv', 'p=0.6'], ['results/crossover/UNIFORM_p07-results.csv', 'p=0.7'], ['results/crossover/UNIFORM_p08-results.csv', 'p=0.8'], ['results/crossover/UNIFORM_p09-results.csv', 'p=0.9']], 'fitness', "Uniform cross method probabilities: Fitness", "Uniform probability", 'Fitness (%)')
    crossoverComparison([['results/crossover/UNIFORM_p01-results.csv', 'p=0.1'], ['results/crossover/UNIFORM_p02-results.csv', 'p=0.2'], ['results/crossover/UNIFORM_p03-results.csv', 'p=0.3'], ['results/crossover/UNIFORM_p04-results.csv', 'p=0.4'], ['results/crossover/UNIFORM_p05-results.csv', 'p=0.5'], ['results/crossover/UNIFORM_p06-results.csv', 'p=0.6'], ['results/crossover/UNIFORM_p07-results.csv', 'p=0.7'], ['results/crossover/UNIFORM_p08-results.csv', 'p=0.8'], ['results/crossover/UNIFORM_p09-results.csv', 'p=0.9']], 'executionTime', "Uniform cross method probabilities: Execution Time", "Uniform probability", 'Execution Time (s)')
