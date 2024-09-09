from jsonschema import validate as json_validate
import sys
import json
from src.classes.eve import EVE
from src.classes.random import RandomGenerator

if __name__ == "__main__":
    with open("config/schema.json", "r") as schemaFile, open(sys.argv[1], "r") as configFile:
        schema = json.load(schemaFile)
        config = json.load(configFile)
        json_validate(config, schema)
        print("Config file is valid.")

        RandomGenerator(config["eve"]["seed"])
        eve = EVE(config["eve"])
        benchmark = eve.run(config["game"]["playerClass"], config["game"]["totalPoints"], config["game"]["maxTime"])

        print(benchmark)
        print("Fitness:", benchmark["bestIndividual"].getFitness())

    sys.exit(0)
