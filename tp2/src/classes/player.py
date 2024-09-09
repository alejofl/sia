from abc import ABC, abstractmethod
import numpy as np
from .random import RandomGenerator


class Player(ABC):
    def __init__(self, totalPoints, height, strength, skill, intelligence, courage, physique):
        self.totalPoints = totalPoints
        self.height = height
        self.strength = strength
        self.skill = skill
        self.intelligence = intelligence
        self.courage = courage
        self.physique = physique
        self.normalizePlayer()
        
    def normalizePlayer(self):
        currentPoints = self.strength + self.skill + self.intelligence + self.courage + self.physique
        self.strength = self.strength * self.totalPoints / currentPoints
        self.skill = self.skill * self.totalPoints / currentPoints
        self.intelligence = self.intelligence * self.totalPoints / currentPoints
        self.courage = self.courage * self.totalPoints / currentPoints
        self.physique = self.physique * self.totalPoints / currentPoints
        self.height=1.3 if self.height < 1.3 else 2.0 if self.height > 2.0 else self.height

    def setHeight(self, height):
        self.height = height
        self.normalizePlayer()

    def setStrength(self, strength):
        self.strength = strength
        self.normalizePlayer()

    def setSkill(self, skill):
        self.skill = skill
        self.normalizePlayer()

    def setIntelligence(self, intelligence):
        self.intelligence = intelligence
        self.normalizePlayer()

    def setCourage(self, courage):
        self.courage = courage
        self.normalizePlayer()

    def setPhysique(self, physique):
        self.physique = physique
        self.normalizePlayer()

    def getAttributeTuples(self):
        return [(self.height, self.setHeight), (self.strength, self.setStrength), (self.skill, self.setSkill), (self.intelligence, self.setIntelligence), (self.courage, self.setCourage), (self.physique, self.setPhysique)]
        
    def generateRandomPlayer(playerClass, totalPoints):
        PLAYER_CLASSES = {
            "WARRIOR": Warrior,
            "ARCHER": Archer,
            "GUARDIAN": Guardian,
            "WIZARD": Wizard
        }

        r = RandomGenerator.getInstance().generator
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

    @abstractmethod
    def getName(self):
        pass

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        return self.height == other.height and self.strength == other.strength and self.skill == other.skill and self.intelligence == other.intelligence and self.courage == other.courage and self.physique == other.physique

    def __hash__(self):
        return hash((self.height, self.strength, self.skill, self.intelligence, self.courage, self.physique))
    
    def __str__(self):
        return f"{self.getName()}(\n\tHeight: {self.height},\n\tStrength: {self.strength},\n\tSkill: {self.skill},\n\tIntelligence: {self.intelligence},\n\tCourage: {self.courage},\n\tPhysique: {self.physique}\n)"

    def __repr__(self):
        return self.__str__()


class Warrior(Player):
    def getFitness(self):
        return 0.6 * self.getAttackStat() + 0.4 * self.getDefenseStat()

    def getName(self):
        return "Warrior"


class Archer(Player):
    def getFitness(self):
        return 0.9 * self.getAttackStat() + 0.1 * self.getDefenseStat()

    def getName(self):
        return "Archer"


class Guardian(Player):
    def getFitness(self):
        return 0.1 * self.getAttackStat() + 0.9 * self.getDefenseStat()

    def getName(self):
        return "Guardian"


class Wizard(Player):
    def getFitness(self):
        return 0.8 * self.getAttackStat() + 0.3 * self.getDefenseStat()

    def getName(self):
        return "Wizard"
