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
            "Warrior": Warrior,
            "Archer": Archer,
            "Guardian": Guardian,
            "Wizard": Wizard
        }

        r = np.random.default_rng()
        height = r.uniform(1.3, 2.0)
        attributes = r.random(5)
        attributes = attributes / np.sum(attributes) * totalPoints

        return PLAYER_CLASSES[playerClass](height, *attributes)

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
