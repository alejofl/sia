import pandas as pd
import sys
from src.classes.player import *


TOTAL_POINTS = 200


if __name__ == "__main__":
    results = pd.read_csv(sys.argv[1])
    
    playerClass = results["class"][0]
    height = results["height"].mean()
    strength = results["strength"].mean()
    skill = results["skill"].mean()
    intelligence = results["intelligence"].mean()
    courage = results["courage"].mean()
    physique = results["physique"].mean()
    
    print(playerClass, "{")
    print("Height: ", height)
    print("Strength: ", strength)
    print("Skill: ", skill)
    print("Intelligence: ", intelligence)
    print("Courage: ", courage)
    print("Physique: ", physique)
    print("}")
    
    inputStrength = int(input("Enter strength: "))
    inputSkill = int(input("Enter skill: "))
    inputIntelligence = int(input("Enter intelligence: "))
    inputCourage = int(input("Enter courage: "))
    inputPhysique = int(input("Enter physique: "))
    
    PLAYER_CLASSES = {
        "WARRIOR": Warrior,
        "ARCHER": Archer,
        "GUARDIAN": Guardian,
        "WIZARD": Wizard
    }
    player = PLAYER_CLASSES[playerClass](TOTAL_POINTS, height, inputStrength, inputSkill, inputIntelligence, inputCourage, inputPhysique)
    print("Fitness:", player.getFitness())

    sys.exit(0)
