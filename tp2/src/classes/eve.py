import time
import numpy as np
from .player import Player


class EVE:    
    def __init__(self, config):
        self.config = config

    @staticmethod
    def selectElite(population, k, options = None):
        n = len(population)

        sortedPopulation = sorted(population, key=lambda x: x.getFitness(), reverse=True)
        newPopulation = []

        for i in range(n):
            p = int(np.max(np.ceil((k - i) / n), 0))

            for _ in range(p):
                newPopulation.append(sortedPopulation[i]) 

        return newPopulation

    @staticmethod
    def selectRoulette(population, k, options = None):
        gen = np.random.default_rng()

        totalFitness = np.sum(list(map(lambda x: x.getFitness(), population)))
        relativeFitness = list(map(lambda x: x.getFitness() / totalFitness, population))
        accumulatedFitness = np.cumsum(relativeFitness)
        newPopulation = []

        for _ in range(k):
            r = gen.random()
            i = np.searchsorted(accumulatedFitness, r)
            newPopulation.append(population[i])

        return newPopulation

    @staticmethod
    def selectUniversal(population, k, options = None):
        gen = np.random.default_rng()

        totalFitness = np.sum(list(map(lambda x: x.getFitness(), population)))
        relativeFitness = list(map(lambda x: x.getFitness() / totalFitness, population))
        accumulatedFitness = np.cumsum(relativeFitness)
        newPopulation = []

        for j in range(k):
            r = (gen.random() + j) / k
            i = np.searchsorted(accumulatedFitness, r)
            newPopulation.append(population[i])

        return newPopulation

    @staticmethod
    def selectBoltzmann(population, k, options = None):
        gen = np.random.default_rng()
        temp = options["tCritic"] + (options["t0"] - options["tCritic"]) * np.exp(- options["k"] * options["generation"])

        sortedFitness = sorted(list(map(lambda x: x.getFitness(), population), reverse=True))
        averageExpFitness = np.average(list(map(lambda x: np.exp(x / temp), sortedFitness)))
        pseudoFitness = list(map(lambda x: np.exp(x / temp) / averageExpFitness, sortedFitness))
        totalPseudoFitness = np.sum(pseudoFitness)
        relativePseudoFitness = pseudoFitness / totalPseudoFitness
        accumulatedPseudoFitness = np.cumsum(relativePseudoFitness)
        newPopulation = []

        for _ in range(k):
            r = gen.random()
            i = np.searchsorted(accumulatedPseudoFitness, r)
            newPopulation.append(sortedFitness[i])

        return newPopulation

    @staticmethod
    def selectRanking(population, k, options = None):
        gen = np.random.default_rng()

        n = len(population)
        sortedFitness = sorted(list(map(lambda x: x.getFitness(), population), reverse=True))
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

    @staticmethod
    def selectDeterministicTournament(population, k, options = None):
        gen = np.random.default_rng()

        m = options["m"]
        newPopulation = []

        for _ in range(k):
            subPopulation = gen.choice(population, m, replace=False)
            newPopulation.append(max(subPopulation, key=lambda x: x.getFitness()))

        return newPopulation

    @staticmethod
    def selectProbabilisticTournament(population, k, options = None):
        gen = np.random.default_rng()

        threshold = options["threshold"]
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
    
    @staticmethod
    def onePointCrossover(player1, player2, options = None):
        gen = np.random.default_rng()

        player1Alleles = player1.getalleles()
        player2Alleles = player2.getalleles()

        randomIndex = gen.integers(0, len(player1Alleles))

        return player1.__class__(player1.totalPoints, *player1Alleles[:randomIndex], *player2Alleles[randomIndex:]), player1.__class__(player1.totalPoints, *player2Alleles[:randomIndex], *player1Alleles[randomIndex:])

    @staticmethod
    def twoPointsCrossover(player1, player2, options = None):
        gen = np.random.default_rng()

        player1Alleles = player1.getalleles()
        player2Alleles = player2.getalleles()

        firstIndex, secondIndex = gen.integers(0, len(player1Alleles), 2)

        if firstIndex > secondIndex:
            firstIndex, secondIndex = secondIndex, firstIndex

        return player1.__class__(player1.totalPoints, *player1Alleles[:firstIndex], *player2Alleles[firstIndex:secondIndex], *player1Alleles[secondIndex:]), player1.__class__(player1.totalPoints, *player2Alleles[:firstIndex], *player1Alleles[firstIndex:secondIndex], *player2Alleles[secondIndex:])

    @staticmethod
    def anularCrossover(player1, player2, options = None):
        gen = np.random.default_rng()

        player1Alleles = player1.getalleles()
        player2Alleles = player2.getalleles()
        n = len(player1Alleles)

        index, length = gen.integers(0, n, 2)

        child1 = player1Alleles.copy()
        child2 = player2Alleles.copy()
        for i in range(length):
            child1[(index + i) % n] = player2Alleles[(index + i) % n]
            child2[(index + i) % n] = player1Alleles[(index + i) % n]

        return player1.__class__(player1.totalPoints, *child1), player1.__class__(player1.totalPoints, *child2)

    @staticmethod
    def uniformCrossover(player1, player2, options = None):
        gen = np.random.default_rng()

        player1Alleles = player1.getalleles()
        player2Alleles = player2.getalleles()
        p = options["p"]

        child1 = []
        child2 = []

        for a1, a2 in zip(player1Alleles, player2Alleles):
            rand = gen.random()
            if rand > p:
                child1.append(a1)
                child2.append(a2)
            else:
                child1.append(a2)
                child2.append(a1)

        return player1.__class__(player1.totalPoints, *child1), player1.__class__(player1.totalPoints, *child2)

    @staticmethod
    def genMutation(player, probability):
        gen = np.random.default_rng()
        if gen.random() < probability:
            randomGen = gen.choice(player.getAttributeTuples())
            randomDelta = gen.uniform(0.65, 1.35)
            randomGen[1](randomGen[0] * randomDelta)

    @staticmethod
    def multigenMutation(player, probability):
        gen = np.random.default_rng()
        attributeTuples = player.getAttributeTuples()
        for attr, setter in attributeTuples:
            if gen.random() < probability:
                randomDelta = gen.uniform(0.65, 1.35)
                setter(attr * randomDelta)

    def evaluateMaxGenerationsStopCondition(self, population, options):
        return self.generationCount >= options["maxGenerations"]

    def evaluateStructureStopCondition(self, population, options):
        if self.previousPopulation is None:
            return False

        isPersistent = len(np.intersect1d(self.previousPopulation, population)) / len(population) >= options["percentage"]
        if isPersistent:
            self.persistentStructureCount += 1
        else:
            self.persistentStructureCount = 0

        return self.persistentStructureCount >= options["minRepeatedGenerations"]

    def evaluateContentStopCondition(self, population, options):
        if self.bestIndividual is None:
            return False

        bestIndividual = max(population, key=lambda x: x.getFitness())
        isPersistent = bestIndividual == self.bestIndividual
        if isPersistent:
            self.persistentContentCount += 1
        else:
            self.persistentContentCount = 0

        return self.persistentContentCount >= options["minRepeatedGenerations"]

    def evaluateDeltaStopCondition(self, population, options):
        if self.bestIndividual is None:
            return False

        bestIndividual = max(population, key=lambda x: x.getFitness())
        delta = abs(bestIndividual.getFitness() - self.bestIndividual.getFitness())
        isPersistent = delta <= options["delta"]
        if isPersistent:
            self.persistentDeltaCount += 1
        else:
            self.persistentDeltaCount = 0

        return self.persistentDeltaCount >= options["minRepeatedGenerations"]

    SELECTION_FUNCTIONS = {
        "ELITE": selectElite,
        "ROULETTE": selectRoulette,
        "UNIVERSAL": selectUniversal,
        "BOLTZMANN": selectBoltzmann,
        "RANKING": selectRanking,
        "DETERMINISTIC_TOURNAMENT": selectDeterministicTournament,
        "PROBABILISTIC_TOURNAMENT": selectProbabilisticTournament
    }

    CROSSOVER_FUNCTIONS = {
        "ONE_POINT": onePointCrossover,
        "TWO_POINTS": twoPointsCrossover,
        "ANULAR": anularCrossover,
        "UNIFORM": uniformCrossover
    }

    MUTATION_FUNCTIONS = {
        "GEN": genMutation,
        "MULTIGEN": multigenMutation
    }
    
    def startExecution(self, maxTime):
        self.startTime = time.time()
        self.maxTime = maxTime
        self.generationCount = 1
        self.previousPopulation = None
        self.persistentStructureCount = 0
        self.bestIndividual = None
        self.persistentContentCount = 0
        self.persistentDeltaCount = 0

    def evaluateStopCondition(self, population):
        if time.time() - self.startTime >= self.maxTime:
            return True

        stopCondition = False
        for stopCondition in self.config["stopCondition"]:
            match stopCondition["method"]:
                case "MAX_GENERATIONS":
                    stopCondition = stopCondition and self.evaluateMaxGenerationsStopCondition(population, stopCondition["options"])
                case "STRUCTURE":
                    stopCondition = stopCondition and self.evaluateStructureStopCondition(population, stopCondition["options"])
                case "CONTENT":
                    stopCondition = stopCondition and self.evaluateContentStopCondition(population, stopCondition["options"])
                case "DELTA":
                    stopCondition = stopCondition and self.evaluateDeltaStopCondition(population, stopCondition["options"])

        self.previousPopulation = population
        self.bestIndividual = max(population, key=lambda x: x.getFitness())
        self.generationCount += 1
        return stopCondition

    def run(self, playerClass, totalPoints, maxTime):
        self.startExecution(maxTime)
        population = [Player.generateRandomPlayer(playerClass, totalPoints) for _ in range(self.config["populationSize"])]

        keepRunning = True
        while keepRunning:
            parents = []
            for selectionMethod in self.config["selection"]:
                matesForSelection = int(np.ceil(self.config["matesPerGeneration"] * selectionMethod["percentage"]))
                options = selectionMethod["options"]
                options["generation"] = self.generationCount
                parents.extend(self.SELECTION_FUNCTIONS[selectionMethod["method"]](population, matesForSelection, options))

            children = []
            parentsCount = len(parents)
            i = 0
            while i < parentsCount:
                parent1 = parents[i]
                parent2 = parents[i + 1 if i + 1 < parentsCount else i]
                child1, child2 = self.CROSSOVER_FUNCTIONS[self.config["crossover"]["method"]](parent1, parent2, self.config["crossover"]["options"])
                self.MUTATION_FUNCTIONS[self.config["mutation"]["method"]](child1, self.config["mutation"]["p"])
                self.MUTATION_FUNCTIONS[self.config["mutation"]["method"]](child2, self.config["mutation"]["p"])
                children.append(child1)
                children.append(child2)
                i += 2

            newPopulation = []
            if self.config["pick"]["type"] == "YOUNG_BIASED":
                populationPlacesLeft = self.config["populationSize"]
                if len(children) <= self.config["populationSize"]:
                    newPopulation.extend(children)
                    populationPlacesLeft -= len(children)
                    if populationPlacesLeft > 0:
                        for pickMethod in self.config["pick"]["selectionMethods"]:
                            qty = np.ceil(populationPlacesLeft * pickMethod["percentage"])
                            newPopulation.extend(self.SELECTION_FUNCTIONS[pickMethod["method"]](population, qty, pickMethod["options"]))
                else:
                    for pickMethod in self.config["pick"]["selectionMethods"]:
                        qty = np.ceil(self.config["populationSize"] * pickMethod["percentage"])
                        newPopulation.extend(self.SELECTION_FUNCTIONS[pickMethod["method"]](children, qty, pickMethod["options"]))
            else:
                people = population + children
                for pickMethod in self.config["pick"]["selectionMethods"]:
                    qty = np.ceil(self.config["populationSize"] * pickMethod["percentage"])
                    newPopulation.extend(self.SELECTION_FUNCTIONS[pickMethod["method"]](people, qty, pickMethod["options"]))            

            population = newPopulation[:self.config["populationSize"]]
            keepRunning = not self.evaluateStopCondition(population)

        return {
            "bestIndividual": self.bestIndividual,
            "generationCount": self.generationCount,
            "executionTime": time.time() - self.startTime
        }
