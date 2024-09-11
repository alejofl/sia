from jsonschema import validate as json_validate
import sys
import json
import time
import csv
from src.classes.eve import EVE
from src.classes.random import RandomGenerator
import traceback
import os

def getOutputFileName():
    inputFile = sys.argv[1]
    inputDir, inputFilename = os.path.split(input_file)  
    inputSubdir = os.path.basename(input_dir)
    outputFilename = input_filename.replace(".json", "-results.csv")  

    outputDir = os.path.join("results", input_subdir) 
    os.makedirs(outputDir, exist_ok=True)  
    return os.path.join(outputDir, outputFilename) 

if __name__ == "__main__":

    input_file = sys.argv[1]
    input_dir, input_filename = os.path.split(input_file)  
    input_subdir = os.path.basename(input_dir)
    output_filename = input_filename.replace(".json", "-results.csv")  

    output_dir = os.path.join("results", input_subdir) 
    os.makedirs(output_dir, exist_ok=True)  
    output_file = os.path.join(output_dir, output_filename) 
    
    with open("config/schema.json", "r") as schemaFile, open(sys.argv[1], "r") as configFile, open(getOutputFileName(), "w") as resultsFile:
        schema = json.load(schemaFile)
        config = json.load(configFile)
        json_validate(config, schema)
        print("Config file is valid.")

        RandomGenerator(config["eve"]["seed"])
        writer = csv.DictWriter(resultsFile, fieldnames=["class", "height", "strength", "skill", "intelligence", "courage", "physique", "fitness", "generationCount", "executionTime", "childrenCount", "mutationCount"])
        writer.writeheader()

        eve = EVE(config["eve"])
        for i in range(config["game"]["runs"]):
            print(f"Run {i + 1}/{config['game']['runs']}:", end=" ")
            try:
                benchmark = eve.run(config["game"]["playerClass"], config["game"]["totalPoints"], config["game"]["maxTime"])
                writer.writerow({
                    "class": config["game"]["playerClass"],
                    "height": benchmark["bestIndividual"].height,
                    "strength": benchmark["bestIndividual"].strength,
                    "skill": benchmark["bestIndividual"].skill,
                    "intelligence": benchmark["bestIndividual"].intelligence,
                    "courage": benchmark["bestIndividual"].courage,
                    "physique": benchmark["bestIndividual"].physique,
                    "fitness": benchmark["bestIndividual"].getFitness(),
                    "generationCount": benchmark["generationCount"],
                    "executionTime": benchmark["executionTime"],
                    "childrenCount": benchmark["childrenCount"],
                    "mutationCount": benchmark["mutationCount"]
                })
                print("DONE")
            except:
                
                print("FAILED: ")
                traceback.print_exc()

    sys.exit(0)
