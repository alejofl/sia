import numpy as np
from .player import Player


class EVE:
    def __init__(self, ):
        # TODO
        pass
 
    def selectElite(population, k):
        n = len(population)

        sortedPopulation = sorted(population, key=lambda x: x.getFitness(), reverse=True)
        newPopulation = []

        for i in range(n):
            p = np.max(np.ceil((k - i) / n), 0)

            for _ in range(p):
                newPopulation.append(sortedPopulation[i]) 

        return newPopulation

    def selectRoulette(population, k):
        gen = np.random.default_rng()

        totalFitness = np.sum(population.map(lambda x: x.getFitness()))
        relativeFitness = population.map(lambda x: x.getFitness() / totalFitness)
        accumulatedFitness = np.cumsum(relativeFitness)
        newPopulation = []

        for _ in range(k):
            r = gen.random()
            i = np.searchsorted(accumulatedFitness, r)
            newPopulation.append(population[i])

        return newPopulation

    def selectUniversal(population, k):
        gen = np.random.default_rng()

        totalFitness = np.sum(population.map(lambda x: x.getFitness()))
        relativeFitness = population.map(lambda x: x.getFitness() / totalFitness)
        accumulatedFitness = np.cumsum(relativeFitness)
        newPopulation = []

        for j in range(k):
            r = (gen.random() + j) / k
            i = np.searchsorted(accumulatedFitness, r)
            newPopulation.append(population[i])

        return newPopulation
 
    def selectBoltzmann(population, k):
        # TODO
        pass

    def selectRanking(population, k):
        gen = np.random.default_rng()

        n = len(population)
        sortedFitness = sorted(population.map(lambda x: x.getFitness()), reverse=True)
        pseudoFitness = [(n - i + 1) / n for i in range(n)]
        totalPseudoFitness = np.sum(pseudoFitness)
        relativePseudoFitness = pseudoFitness / totalPseudoFitness
        accumulatedPseudoFitness = np.cumsum(relativePseudoFitness)
        newPopulation = []

        for _ in range(k):
            r = gen.random()
            i = np.searchsorted(accumulatedPseudoFitness, r)
            newPopulation.append(sortedFitness[i])

        return newPopulation

    def selectDeterministicTournament(population, k):
        gen = np.random.default_rng()

        m = len(population) // 3 # TODO: check if this is correct
        newPopulation = []

        for _ in range(k):
            subPopulation = gen.choice(population, m, replace=False)
            newPopulation.append(max(subPopulation, key=lambda x: x.getFitness()))

        return newPopulation

    def selectProbabilisticTournament(population, k):
        gen = np.random.default_rng()

        threshold = 0.7 # TODO: check if this is correct
        newPopulation = []

        for _ in range(k):
            player1, player2 = gen.choice(population, 2, replace=False)
            if player1.getFitness() > player2.getFitness():
                player1, player2 = player2, player1

            r = gen.random()
            if r < threshold:
                newPopulation.append(player2)
            else:
                newPopulation.append(player1)

        return newPopulation
 
    # poblacion = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    # padres = [1, 2, 3, 4, 5, 6, 7]
    #                 |
    # hijos = [1-2, 2-7, 3-1, ...]
    def run(self, initialPopulation, playerClass, totalPoints, maxTime):
        population = [Player.generateRandomPlayer(playerClass, totalPoints) for _ in range(initialPopulation)]
        
        # Select parents
        # Generate children, with crossover and mutation
        # Pick new population
        # Repeat
