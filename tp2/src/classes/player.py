from abc import ABC, abstractmethod
import numpy as np


class Player(ABC):
    def __init__(self, totalPoints, height, strength, skill, intelligence, courage, physique):
        self.totalPoints = totalPoints
        self.height = height

        currentPoints = strength + skill + intelligence + courage + physique
        self.strength = strength * totalPoints / currentPoints
        self.skill = skill * totalPoints / currentPoints
        self.intelligence = intelligence * totalPoints / currentPoints
        self.courage = courage * totalPoints / currentPoints
        self.physique = physique * totalPoints / currentPoints
        
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

        return PLAYER_CLASSES[playerClass](totalPoints, height, *attributes)

    def getalleles(self):
        return [self.height, self.strength, self.skill, self.intelligence, self.courage, self.physique]

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

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        return self.height == other.height and self.strength == other.strength and self.skill == other.skill and self.intelligence == other.intelligence and self.courage == other.courage and self.physique == other.physique

    def __hash__(self):
        return hash((self.height, self.strength, self.skill, self.intelligence, self.courage, self.physique))


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
