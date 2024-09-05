from abc import ABC, abstractmethod
import numpy as np


class Player(ABC):
    def __init__(self, height, strength, skill, intelligence, courage, physique):
        self.height = height
        self.strength = strength
        self.skill = skill
        self.intelligence = intelligence
        self.courage = courage
        self.physique = physique
        
    def generateRandomPlayer(playerClass, totalPoints):
        PLAYER_CLASSES = {
            "WARRIOR": Warrior,
            "ARCHER": Archer,
            "GUARDIAN": Guardian,
            "WIZARD": Wizard
        }

        r = np.random.default_rng()
        height = r.uniform(1.3, 2.0)
        attributes = r.random(5)
        attributes = attributes / np.sum(attributes) * totalPoints

        return PLAYER_CLASSES[playerClass](height, *attributes)

    def getalleles(self):
        return [self.height, self.strength, self.skill, self.intelligence, self.courage, self.physique]

    def onePointCrossover(self, player2, options = None):
        gen = np.random.default_rng()

        player1Alleles = self.getalleles()
        player2Alleles = player2.getalleles()

        randomIndex = gen.integers(0, len(player1Alleles))

        return self.__class__(*player1Alleles[:randomIndex], *player2Alleles[randomIndex:]), self.__class__(*player2Alleles[:randomIndex], *player1Alleles[randomIndex:])

    def twoPointsCrossover(self, player2, options = None):
        gen = np.random.default_rng()

        player1Alleles = self.getalleles()
        player2Alleles = player2.getalleles()

        firstIndex, secondIndex = gen.integers(0, len(player1Alleles), 2)

        if firstIndex > secondIndex:
            firstIndex, secondIndex = secondIndex, firstIndex

        return self.__class__(*player1Alleles[:firstIndex], *player2Alleles[firstIndex:secondIndex], *player1Alleles[secondIndex:]), self.__class__(*player2Alleles[:firstIndex], *player1Alleles[firstIndex:secondIndex], *player2Alleles[secondIndex:])

    def anularCrossover(self, player2, options = None):
        gen = np.random.default_rng()

        player1Alleles = self.getalleles()
        player2Alleles = player2.getalleles()
        n = len(player1Alleles)

        index, length = gen.integers(0, n, 2)

        child1 = player1Alleles.copy()
        child2 = player2Alleles.copy()
        for i in range(length):
            child1[(index + i) % n] = player2Alleles[(index + i) % n]
            child2[(index + i) % n] = player1Alleles[(index + i) % n]

        return self.__class__(*child1), self.__class__(*child2)

    def uniformCrossover(self, player2, options = None):
        gen = np.random.default_rng()

        player1Alleles = self.getalleles()
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

        return self.__class__(*child1), self.__class__(*child2)

    def getNormStrength(self):
        return 100 * np.tanh(0.01 * self.strength)

    def getNormSkill(self):
        return np.tanh(0.01 * self.skill)

    def getNormIntelligence(self):
        return 0.6 * np.tanh(0.01 * self.intelligence)

    def getNormCourage(self):
        return np.tanh(0.01 * self.courage)

    def getNormPhysique(self):
        return 100 * np.tanh(0.01 * self.physique)

    def getAttackStat(self):
        atm = 0.5 - (3 * self.height - 5) ** 4 + (3 * self.height - 5) ** 2 + self.height / 2
        return (self.getNormSkill() + self.getNormIntelligence()) * self.getNormStrength() * atm

    def getDefenseStat(self):
        dem = 2 + (3 * self.height - 5) ** 4 - (3 * self.height - 5) ** 2 - self.height / 2
        return (self.getNormCourage() + self.getNormIntelligence()) * self.getNormPhysique() * dem

    @abstractmethod
    def getFitness(self):
        pass


class Warrior(Player):
    def getFitness(self):
        return 0.6 * self.getAttackStat() + 0.4 * self.getDefenseStat()


class Archer(Player):
    def getFitness(self):
        return 0.9 * self.getAttackStat() + 0.1 * self.getDefenseStat()


class Guardian(Player):
    def getFitness(self):
        return 0.1 * self.getAttackStat() + 0.9 * self.getDefenseStat()


class Wizard(Player):
    def getFitness(self):
        return 0.8 * self.getAttackStat() + 0.3 * self.getDefenseStat()
