import pandas as pd
import glob
import matplotlib.pyplot as plt
import scipy.stats as ss 
import matplotlib.ticker as mticker

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
            plt.gca().yaxis.set_major_formatter(mticker.FormatStrFormatter('%.5f'))


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

    #Diapo 14
    # comparison([['results/mutation/gen-0,1-results.csv', 'p=0.1'], ['results/mutation/gen-0,2-results.csv', 'p=0.2'],['results/mutation/gen-0,3-results.csv', 'p=0.3'], ['results/mutation/gen-0,4-results.csv', 'p=0.4'],['results/mutation/gen-0,5-results.csv', 'p=0.5'], ['results/mutation/gen-0,6-results.csv', 'p=0.6'],['results/mutation/gen-0,7-results.csv', 'p=0.7'], ['results/mutation/gen-0,8-results.csv', 'p=0.8'],['results/mutation/gen-0,9-results.csv', 'p=0.9']], 'fitness',"Gen mutation method probabilities: Fitness", "Mutation probability", 'Fitness (%)')
    # comparison([['results/mutation/gen-0,1-results.csv', 'p=0.1'], ['results/mutation/gen-0,2-results.csv', 'p=0.2'],['results/mutation/gen-0,3-results.csv', 'p=0.3'], ['results/mutation/gen-0,4-results.csv', 'p=0.4'],['results/mutation/gen-0,5-results.csv', 'p=0.5'], ['results/mutation/gen-0,6-results.csv', 'p=0.6'],['results/mutation/gen-0,7-results.csv', 'p=0.7'], ['results/mutation/gen-0,8-results.csv', 'p=0.8'],['results/mutation/gen-0,9-results.csv', 'p=0.9']], 'executionTime',"Gen mutation method probabilities: Execution Time", "Mutation probability", 'Execution Time (s)')
    # comparison([['results/mutation/gen-0,1-results.csv', 'p=0.1'], ['results/mutation/gen-0,2-results.csv', 'p=0.2'],['results/mutation/gen-0,3-results.csv', 'p=0.3'], ['results/mutation/gen-0,4-results.csv', 'p=0.4'],['results/mutation/gen-0,5-results.csv', 'p=0.5'], ['results/mutation/gen-0,6-results.csv', 'p=0.6'],['results/mutation/gen-0,7-results.csv', 'p=0.7'], ['results/mutation/gen-0,8-results.csv', 'p=0.8'],['results/mutation/gen-0,9-results.csv', 'p=0.9']], 'mutationCount',"Gen mutation method probabilities: Mutation count", "Mutation probability", 'Mutations')

    #Diapo 15
    # comparison([['results/mutation/multigen-0,1-results.csv', 'p=0.1'], ['results/mutation/multigen-0,2-results.csv', 'p=0.2'],['results/mutation/multigen-0,3-results.csv', 'p=0.3'], ['results/mutation/multigen-0,4-results.csv', 'p=0.4'],['results/mutation/multigen-0,5-results.csv', 'p=0.5'], ['results/mutation/multigen-0,6-results.csv', 'p=0.6'],['results/mutation/multigen-0,7-results.csv', 'p=0.7'], ['results/mutation/multigen-0,8-results.csv', 'p=0.8'],['results/mutation/multigen-0,9-results.csv', 'p=0.9']], 'fitness',"Multigen mutation method probabilities: Fitness", "Mutation probability", 'Fitness (%)')
    # comparison([['results/mutation/multigen-0,1-results.csv', 'p=0.1'], ['results/mutation/multigen-0,2-results.csv', 'p=0.2'],['results/mutation/multigen-0,3-results.csv', 'p=0.3'], ['results/mutation/multigen-0,4-results.csv', 'p=0.4'],['results/mutation/multigen-0,5-results.csv', 'p=0.5'], ['results/mutation/multigen-0,6-results.csv', 'p=0.6'],['results/mutation/multigen-0,7-results.csv', 'p=0.7'], ['results/mutation/multigen-0,8-results.csv', 'p=0.8'],['results/mutation/multigen-0,9-results.csv', 'p=0.9']], 'executionTime',"Multigen mutation method probabilities: Execution Time", "Mutation probability", 'Execution Time (s)')
    # comparison([['results/mutation/multigen-0,1-results.csv', 'p=0.1'], ['results/mutation/multigen-0,2-results.csv', 'p=0.2'],['results/mutation/multigen-0,3-results.csv', 'p=0.3'], ['results/mutation/multigen-0,4-results.csv', 'p=0.4'],['results/mutation/multigen-0,5-results.csv', 'p=0.5'], ['results/mutation/multigen-0,6-results.csv', 'p=0.6'],['results/mutation/multigen-0,7-results.csv', 'p=0.7'], ['results/mutation/multigen-0,8-results.csv', 'p=0.8'],['results/mutation/multigen-0,9-results.csv', 'p=0.9']], 'mutationCount',"Multigen mutation method probabilities: Mutation count", "Mutation probability", 'Mutations')
    
    #Diapo 16
    # comparison([['results/mutation/gen-0,7-results.csv', 'Gen (p=0.7)'], ['results/mutation/gen-0,8-results.csv', 'Gen (p=0.8)'], ['results/mutation/multigen-0,1-results.csv', 'Multigen (p=0.1)'], ['results/mutation/multigen-0,2-results.csv', 'Multigen (p=0.2)']], 'fitness',"Gen vs Multigen mutation: Best fitness", "Mutation method", fitnessLabel)
    # comparison([['results/mutation/gen-0,7-results.csv', 'Gen (p=0.7)'], ['results/mutation/gen-0,8-results.csv', 'Gen (p=0.8)'], ['results/mutation/multigen-0,1-results.csv', 'Multigen (p=0.1)'], ['results/mutation/multigen-0,2-results.csv', 'Multigen (p=0.2)']], 'executionTime',"Gen vs Multigen mutation: Execution time", "Mutation method", executionTimeLabel)

    #Diapo 18
    # comparison([['results/selection/probTournament-0,5-results.csv', 'th=0.5'],['results/selection/probTournament-0,6-results.csv', 'th=0.6'],['results/selection/probTournament-0,7-results.csv', 'th=0.7'],['results/selection/probTournament-0,8-results.csv', 'th=0.8'],['results/selection/probTournament-0,9-results.csv', 'th=0.9'], ], 'fitness',"Probabilistic Tournament selection method probabilities: best fitness", "Threshold", fitnessLabel)
    # comparison([['results/selection/probTournament-0,5-results.csv', 'th=0.5'],['results/selection/probTournament-0,6-results.csv', 'th=0.6'],['results/selection/probTournament-0,7-results.csv', 'th=0.7'],['results/selection/probTournament-0,8-results.csv', 'th=0.8'],['results/selection/probTournament-0,9-results.csv', 'th=0.9'], ], 'executionTime',"Probabilistic Tournament selection method probabilities: execution time", "Threshold", executionTimeLabel)

    #Diapo 19
    # comparison([['results/selection/detTournament-2,2%-results.csv', 'm=22'],['results/selection/detTournament-2,4%-results.csv', 'm=24'],['results/selection/detTournament-2,5%-results.csv', 'm=25'],['results/selection/detTournament-2,6%-results.csv', 'm=26'],['results/selection/detTournament-2,7%-results.csv', 'm=27'],['results/selection/detTournament-5%-results.csv', 'm=50'], ['results/selection/detTournament-7,5%-results.csv', 'm=75'],['results/selection/detTournament-10%-results.csv', 'm=100'],['results/selection/detTournament-25%-results.csv', 'm=250'],['results/selection/detTournament-50%-results.csv', 'm=500'],['results/selection/detTournament-75%-results.csv', 'm=750'] ], 'fitness',"Deterministic Tournament selection method probabilities: best fitness", "Mates amount", fitnessLabel)
    # comparison([['results/selection/detTournament-2,2%-results.csv', 'm=22'],['results/selection/detTournament-2,4%-results.csv', 'm=24'],['results/selection/detTournament-2,5%-results.csv', 'm=25'],['results/selection/detTournament-2,6%-results.csv', 'm=26'],['results/selection/detTournament-2,7%-results.csv', 'm=27'],['results/selection/detTournament-5%-results.csv', 'm=50'], ['results/selection/detTournament-7,5%-results.csv', 'm=75'],['results/selection/detTournament-10%-results.csv', 'm=100'],['results/selection/detTournament-25%-results.csv', 'm=250'],['results/selection/detTournament-50%-results.csv', 'm=500'],['results/selection/detTournament-75%-results.csv', 'm=750'] ], 'executionTime',"Deterministic Tournament selection method probabilities: execution time", "Mates amount", executionTimeLabel)
    
    #Diapo 20
    # comparison([['results/selection/boltzmann/t-critic/boltzmann-tc0,00-results.csv', '0,00'],['results/selection/boltzmann/t-critic/boltzmann-tc0,01-results.csv', '0,01'],['results/selection/boltzmann/t-critic/boltzmann-tc0,02-results.csv', '0,02'],['results/selection/boltzmann/t-critic/boltzmann-tc0,03-results.csv', '0,03'],['results/selection/boltzmann/t-critic/boltzmann-tc0,04-results.csv', '0,04'],['results/selection/boltzmann/t-critic/boltzmann-tc0,05-results.csv', '0,05'],['results/selection/boltzmann/t-critic/boltzmann-tc0,06-results.csv', '0,06'],['results/selection/boltzmann/t-critic/boltzmann-tc0,07-results.csv', '0,07'],['results/selection/boltzmann/t-critic/boltzmann-tc0,08-results.csv', '0,08'],['results/selection/boltzmann/t-critic/boltzmann-tc0,09-results.csv', '0,09'],['results/selection/boltzmann/t-critic/boltzmann-tc0,10-results.csv', '0,10'] ], 'fitness',"Selection Boltzmann (T-Critic): fitness", "T-Critic", fitnessLabel)
    # comparison([['results/selection/boltzmann/t-critic/boltzmann-tc0,00-results.csv', '0,00'],['results/selection/boltzmann/t-critic/boltzmann-tc0,01-results.csv', '0,01'],['results/selection/boltzmann/t-critic/boltzmann-tc0,02-results.csv', '0,02'],['results/selection/boltzmann/t-critic/boltzmann-tc0,03-results.csv', '0,03'],['results/selection/boltzmann/t-critic/boltzmann-tc0,04-results.csv', '0,04'],['results/selection/boltzmann/t-critic/boltzmann-tc0,05-results.csv', '0,05'],['results/selection/boltzmann/t-critic/boltzmann-tc0,06-results.csv', '0,06'],['results/selection/boltzmann/t-critic/boltzmann-tc0,07-results.csv', '0,07'],['results/selection/boltzmann/t-critic/boltzmann-tc0,08-results.csv', '0,08'],['results/selection/boltzmann/t-critic/boltzmann-tc0,09-results.csv', '0,09'],['results/selection/boltzmann/t-critic/boltzmann-tc0,10-results.csv', '0,10'] ], 'executionTime',"Selection Boltzmann (T-Critic): Execution Time", "T-Critic", executionTimeLabel)
    
    #Diapo 21
    # comparison([['results/selection/boltzmann/t0/boltzmann-t0-10-results.csv', '10'],['results/selection/boltzmann/t0/boltzmann-t0-20-results.csv', '20'],['results/selection/boltzmann/t0/boltzmann-t0-30-results.csv', '30'],['results/selection/boltzmann/t0/boltzmann-t0-40-results.csv', '40'],['results/selection/boltzmann/t0/boltzmann-t0-50-results.csv', '50'],['results/selection/boltzmann/t0/boltzmann-t0-60-results.csv', '60'],['results/selection/boltzmann/t0/boltzmann-t0-70-results.csv', '70'],['results/selection/boltzmann/t0/boltzmann-t0-80-results.csv', '80'],['results/selection/boltzmann/t0/boltzmann-t0-90-results.csv', '90'],['results/selection/boltzmann/t0/boltzmann-t0-100-results.csv', '100'] ], 'fitness',"Selection Boltzmann (t0): Fitness", "t0", fitnessLabel)
    # comparison([['results/selection/boltzmann/t0/boltzmann-t0-10-results.csv', '10'],['results/selection/boltzmann/t0/boltzmann-t0-20-results.csv', '20'],['results/selection/boltzmann/t0/boltzmann-t0-30-results.csv', '30'],['results/selection/boltzmann/t0/boltzmann-t0-40-results.csv', '40'],['results/selection/boltzmann/t0/boltzmann-t0-50-results.csv', '50'],['results/selection/boltzmann/t0/boltzmann-t0-60-results.csv', '60'],['results/selection/boltzmann/t0/boltzmann-t0-70-results.csv', '70'],['results/selection/boltzmann/t0/boltzmann-t0-80-results.csv', '80'],['results/selection/boltzmann/t0/boltzmann-t0-90-results.csv', '90'],['results/selection/boltzmann/t0/boltzmann-t0-100-results.csv', '100'] ], 'executionTime',"Selection Boltzmann (t0): Execution time", "t0", executionTimeLabel)
    
    #Diapo 22
    # comparison([['results/selection/boltzmann/k/boltzmann-k-0,001-results.csv', '0,001'],['results/selection/boltzmann/k/boltzmann-k-0,010-results.csv', '0,01'],['results/selection/boltzmann/k/boltzmann-k-0,020-results.csv', '0,02'],['results/selection/boltzmann/k/boltzmann-k-0,030-results.csv', '0,03'],['results/selection/boltzmann/k/boltzmann-k-0,040-results.csv', '0,04'],['results/selection/boltzmann/k/boltzmann-k-0,050-results.csv', '0,05'],['results/selection/boltzmann/k/boltzmann-k-0,060-results.csv', '0,06'],['results/selection/boltzmann/k/boltzmann-k-0,070-results.csv', '0,07'],['results/selection/boltzmann/k/boltzmann-k-0,080-results.csv', '0,08'],['results/selection/boltzmann/k/boltzmann-k-0,090-results.csv', '0,09'],['results/selection/boltzmann/k/boltzmann-k-0,100-results.csv', '0,1'] ], 'fitness',"Selection Boltzmann (k): Fitness", "k", fitnessLabel)
    # comparison([['results/selection/boltzmann/k/boltzmann-k-0,001-results.csv', '0,001'],['results/selection/boltzmann/k/boltzmann-k-0,010-results.csv', '0,01'],['results/selection/boltzmann/k/boltzmann-k-0,020-results.csv', '0,02'],['results/selection/boltzmann/k/boltzmann-k-0,030-results.csv', '0,03'],['results/selection/boltzmann/k/boltzmann-k-0,040-results.csv', '0,04'],['results/selection/boltzmann/k/boltzmann-k-0,050-results.csv', '0,05'],['results/selection/boltzmann/k/boltzmann-k-0,060-results.csv', '0,06'],['results/selection/boltzmann/k/boltzmann-k-0,070-results.csv', '0,07'],['results/selection/boltzmann/k/boltzmann-k-0,080-results.csv', '0,08'],['results/selection/boltzmann/k/boltzmann-k-0,090-results.csv', '0,09'],['results/selection/boltzmann/k/boltzmann-k-0,100-results.csv', '0,1'] ], 'executionTime',"Selection Boltzmann (k): Execution Time", "k", executionTimeLabel)
    
    #Diapo 23
    # comparison([['results/selection/elite-results.csv', 'ELI'],['results/selection/roulette-results.csv', 'ROU'],['results/selection/universal-results.csv', 'UNI'],['results/selection/boltzmann-results.csv', 'BOL'],['results/selection/ranking-results.csv', 'RAN'],['results/selection/detTournament-2,5%-results.csv', 'DET'],['results/selection/probTournament-0,6-results.csv', 'PRO'], ], 'fitness',"All selection method probabilities: best fitness", "Method", fitnessLabel)
    # comparison([['results/selection/elite-results.csv', 'ELI'],['results/selection/roulette-results.csv', 'ROU'],['results/selection/universal-results.csv', 'UNI'],['results/selection/boltzmann/k/boltzmann-k-0,090-results.csv', 'BOL'],['results/selection/ranking-results.csv', 'RAN'],['results/selection/detTournament-2,5%-results.csv', 'DET'],['results/selection/probTournament-0,6-results.csv', 'PRO'], ], 'executionTime',"All selection method probabilities: execution time", "Method", executionTimeLabel)
    
    #Diapo 25
    # comparison([['results/selection/elite-results.csv', 'ELI'],['results/twoMethods/elite-boltzmann-results.csv', 'ELI-BOL'],['results/twoMethods/elite-detTournament-results.csv', 'ELI-DET'],['results/twoMethods/elite-probTournament-results.csv', 'ELI-PRO'],['results/twoMethods/elite-ranking-results.csv', 'ELI-RAN'],['results/twoMethods/elite-roulette-results.csv', 'ELI-ROU'],['results/twoMethods/elite-universal-results.csv', 'ELI-UNI']], 'fitness',"All selection combined with Elite probabilities: best fitness", "Method", fitnessLabel)
    # comparison([['results/selection/elite-results.csv', 'ELI'],['results/twoMethods/elite-boltzmann-results.csv', 'ELI-BOL'],['results/twoMethods/elite-detTournament-results.csv', 'ELI-DET'],['results/twoMethods/elite-probTournament-results.csv', 'ELI-PRO'],['results/twoMethods/elite-ranking-results.csv', 'ELI-RAN'],['results/twoMethods/elite-roulette-results.csv', 'ELI-ROU'],['results/twoMethods/elite-universal-results.csv', 'ELI-UNI']], 'executionTime',"All selection combined with Elite probabilities: execution time", "Method", executionTimeLabel)
    
    #Diapo 27
    # comparison([['results/mutation/gen-0,1-noTime-results.csv', 'p=0.1'], ['results/mutation/gen-0,2-noTime-results.csv', 'p=0.2'],['results/mutation/gen-0,3-noTime-results.csv', 'p=0.3'], ['results/mutation/gen-0,4-noTime-results.csv', 'p=0.4'],['results/mutation/gen-0,5-noTime-results.csv', 'p=0.5'], ['results/mutation/gen-0,6-noTime-results.csv', 'p=0.6'],['results/mutation/gen-0,7-noTime-results.csv', 'p=0.7'], ['results/mutation/gen-0,8-noTime-results.csv', 'p=0.8'],['results/mutation/gen-0,9-noTime-results.csv', 'p=0.9']], 'fitness',"Gen mutation method probabilities: Fitness", "Mutation probability", 'Fitness (%)')
    # comparison([['results/mutation/multigen-0,1-noTime-results.csv', 'p=0.1'], ['results/mutation/multigen-0,2-noTime-results.csv', 'p=0.2'], ['results/mutation/multigen-0,3-noTime-results.csv', 'p=0.3'], ['results/mutation/multigen-0,4-noTime-results.csv', 'p=0.4'], ['results/mutation/multigen-0,5-noTime-results.csv', 'p=0.5'], ['results/mutation/multigen-0,6-noTime-results.csv', 'p=0.6'], ['results/mutation/multigen-0,7-noTime-results.csv', 'p=0.7'], ['results/mutation/multigen-0,8-noTime-results.csv', 'p=0.8'], ['results/mutation/multigen-0,9-noTime-results.csv', 'p=0.9']], 'fitness',"Multigen mutation method probabilities: Fitness", "Mutation probability", 'Fitness (%)')
    
    #Diapo 30
    # comparison([['results/selection/noTime/elite-notime-results.csv', 'ELI'],['results/selection/noTime/roulette-notime-results.csv', 'ROU'],['results/selection/noTime/universal-notime-results.csv', 'UNI'],['results/selection/noTime/boltzmann-notime-results.csv', 'BOL'],['results/selection/noTime/ranking-notime-results.csv', 'RAN'],['results/selection/noTime/detTournament-notime-results.csv', 'DET'],['results/selection/noTime/probTournament-notime-results.csv', 'PRO'], ], 'fitness',"All selection method probabilities: best fitness", "Method", fitnessLabel)

    #Diapo 29
    # comparison([['results/pick/traditional-results.csv', 'Traditional'],['results/pick/young_biased-results.csv', 'Young Bias']], 'fitness', "Pick type: fitness", "Pick type", fitnessLabel)
    # comparison([['results/pick/traditional-results.csv', 'Traditional'],['results/pick/young_biased-results.csv', 'Young Bias']], 'executionTime', "Pick type: Execution time", "Pick type", executionTimeLabel)

    #Diapo 30
    # comparison([['results/mates/mates-10-results.csv', '10%'],['results/mates/mates-30-results.csv', '30%'], ['results/mates/mates-50-results.csv', '50%'], ['results/mates/mates-70-results.csv', '70%'], ['results/mates/mates-100-results.csv', '100%']], 'fitness', "Mates per generation: fitness", "Individuals (%)", fitnessLabel)
    # comparison([['results/mates/mates-10-results.csv', '10%'],['results/mates/mates-30-results.csv', '30%'], ['results/mates/mates-50-results.csv', '50%'], ['results/mates/mates-70-results.csv', '70%'], ['results/mates/mates-100-results.csv', '100%']], 'executionTime', "Mates per generation: Execution time", "Individuals (%)", executionTimeLabel)    


    # Ejercicio 1.2

    # Diapo 44: mutación - gen

    # comparison([['results/ej1.2/2.mutation/gen-0,1-results.csv', 'p=0.1'],['results/ej1.2/2.mutation/gen-0,2-results.csv', 'p=0.2'], ['results/ej1.2/2.mutation/gen-0,3-results.csv', 'p=0.3'], ['results/ej1.2/2.mutation/gen-0,4-results.csv', 'p=0.4'], ['results/ej1.2/2.mutation/gen-0,5-results.csv', 'p=0.5'], ['results/ej1.2/2.mutation/gen-0,6-results.csv', 'p=0.6'], ['results/ej1.2/2.mutation/gen-0,7-results.csv', 'p=0.7'], ['results/ej1.2/2.mutation/gen-0,8-results.csv', 'p=0.8'], ['results/ej1.2/2.mutation/gen-0,9-results.csv', 'p=0.9']], 'fitness',"Gen mutation method probabilities: Fitness", "Mutation probability", fitnessLabel)
    # comparison([['results/ej1.2/2.mutation/gen-0,1-results.csv', 'p=0.1'],['results/ej1.2/2.mutation/gen-0,2-results.csv', 'p=0.2'], ['results/ej1.2/2.mutation/gen-0,3-results.csv', 'p=0.3'], ['results/ej1.2/2.mutation/gen-0,4-results.csv', 'p=0.4'], ['results/ej1.2/2.mutation/gen-0,5-results.csv', 'p=0.5'], ['results/ej1.2/2.mutation/gen-0,6-results.csv', 'p=0.6'], ['results/ej1.2/2.mutation/gen-0,7-results.csv', 'p=0.7'], ['results/ej1.2/2.mutation/gen-0,8-results.csv', 'p=0.8'], ['results/ej1.2/2.mutation/gen-0,9-results.csv', 'p=0.9']], 'executionTime',"Gen mutation method probabilities: Execution time", "Mutation probability", executionTimeLabel)

    # # Diapo 45: mutación - multigen

    # comparison([['results/ej1.2/2.mutation/multigen-0,1-results.csv', 'p=0.1'],['results/ej1.2/2.mutation/multigen-0,2-results.csv', 'p=0.2'], ['results/ej1.2/2.mutation/multigen-0,3-results.csv', 'p=0.3'], ['results/ej1.2/2.mutation/multigen-0,4-results.csv', 'p=0.4'], ['results/ej1.2/2.mutation/multigen-0,5-results.csv', 'p=0.5'], ['results/ej1.2/2.mutation/multigen-0,6-results.csv', 'p=0.6'], ['results/ej1.2/2.mutation/multigen-0,7-results.csv', 'p=0.7'], ['results/ej1.2/2.mutation/multigen-0,8-results.csv', 'p=0.8'], ['results/ej1.2/2.mutation/multigen-0,9-results.csv', 'p=0.9']], 'fitness',"Multigen mutation method probabilities: Fitness", "Mutation probability", fitnessLabel)
    # comparison([['results/ej1.2/2.mutation/multigen-0,1-results.csv', 'p=0.1'],['results/ej1.2/2.mutation/multigen-0,2-results.csv', 'p=0.2'], ['results/ej1.2/2.mutation/multigen-0,3-results.csv', 'p=0.3'], ['results/ej1.2/2.mutation/multigen-0,4-results.csv', 'p=0.4'], ['results/ej1.2/2.mutation/multigen-0,5-results.csv', 'p=0.5'], ['results/ej1.2/2.mutation/multigen-0,6-results.csv', 'p=0.6'], ['results/ej1.2/2.mutation/multigen-0,7-results.csv', 'p=0.7'], ['results/ej1.2/2.mutation/multigen-0,8-results.csv', 'p=0.8'], ['results/ej1.2/2.mutation/multigen-0,9-results.csv', 'p=0.9']], 'executionTime',"Multigen mutation method probabilities: Execution time", "Mutation probability", executionTimeLabel)

    # #Diapo 41
    # comparison([['results/ej1.2/2.mutation/gen-0,8-results.csv', 'Gen (p=0.8)'], ['results/ej1.2/2.mutation/multigen-0,1-results.csv', 'Multigen (p=0.1)']], 'fitness',"Gen vs Multigen mutation: Best fitness", "Mutation method", fitnessLabel)
    # comparison([['results/ej1.2/2.mutation/gen-0,8-results.csv', 'Gen (p=0.8)'], ['results/ej1.2/2.mutation/multigen-0,1-results.csv', 'Multigen (p=0.1)']], "executionTime", "Gen vs Multigen mutation: Execution time", "Mutation method", executionTimeLabel)

    comparison([['results/ej1.2/3.selection/elite-results.csv', 'ELI'],['results/ej1.2/3.selection/roulette-results.csv', 'ROU'],['results/ej1.2/3.selection/universal-results.csv', 'UNI'],['results/ej1.2/3.selection/boltzmann-results.csv', 'BOL'],['results/ej1.2/3.selection/ranking-results.csv', 'RAN'],['results/ej1.2/3.selection/detTournament-results.csv', 'DET'],['results/ej1.2/3.selection/probTournament-results.csv', 'PRO']], 'fitness',"All selection methods: Fitness", "Method", fitnessLabel)
    comparison([['results/ej1.2/3.selection/elite-results.csv', 'ELI'],['results/ej1.2/3.selection/roulette-results.csv', 'ROU'],['results/ej1.2/3.selection/universal-results.csv', 'UNI'],['results/ej1.2/3.selection/boltzmann-results.csv', 'BOL'],['results/ej1.2/3.selection/ranking-results.csv', 'RAN'],['results/ej1.2/3.selection/detTournament-results.csv', 'DET'],['results/ej1.2/3.selection/probTournament-results.csv', 'PRO']], 'executionTime',"All selection methods: Execution time", "Method", executionTimeLabel)