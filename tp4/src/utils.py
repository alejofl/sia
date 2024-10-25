import csv
import os
import sys
import numpy as np
import pandas as pd
from .constants import Constants


class Utils:
    EUROPE_DATASET_PATH = os.path.join(os.path.dirname(sys.argv[0]), "resources", "europe.csv")
    LETTERS_DATASET_PATH = os.path.join(os.path.dirname(sys.argv[0]), "resources", "letters.txt")

    @staticmethod
    def getEuropeDataset(onlyData=False):
        df = pd.read_csv(Utils.EUROPE_DATASET_PATH)
        df = Utils.zScoreData(df)
        countries = df["Country"].to_numpy()
        data = df.reset_index().drop(columns=["Country"])
        if onlyData:
            return data.drop(columns=["index"]).to_numpy()
        return countries, data.to_numpy()
    
    @staticmethod
    def getLettersDataset():
        with open(Utils.LETTERS_DATASET_PATH, "r") as file:
            lines = file.readlines()
            letters = []
            letter = []
            for i, line in enumerate(lines):
                letter.append([int(x) for x in line.split()])
                if i % 5 == 4:
                    letters.append(np.array(letter).flatten())
                    letter = []
            return letters

    @staticmethod
    def zScoreData(df):
        def zScore(col):
            mean = col.mean()
            std = col.std()
            return col.apply(lambda x: (x - mean) / std)

        return df.apply(lambda col: zScore(col) if col.name != "Country" else col)

    @staticmethod
    def saveKohonenOutput(filename, kohonen, countries, dataset):
        filepath = os.path.join(os.path.dirname(sys.argv[0]), "results", "kohonen", filename)
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w") as file:
            writer = csv.writer(file)
            writer.writerow([*pd.read_csv(Utils.EUROPE_DATASET_PATH, nrows=0).columns.tolist(), "row", "col"])
            for country, data in zip(countries, dataset):
                winnerNeuron = kohonen.test(data)
                writer.writerow([country, *data[1:], *winnerNeuron])

    @staticmethod
    def saltAndPepper(data, noiseLevel):
        random = Constants.getInstance().random
        noisyData = np.copy(data)
        for i in range(len(noisyData)):
            if random.random() < noiseLevel:
                noisyData[i] = -noisyData[i]
        return noisyData
    
    @staticmethod
    def checkOrtogonality(letters, representations):
        values = []
        for l in letters:
            values.append(representations[ord(l) - ord("A")])
        values = np.array(values)
        ortoMatrix = np.dot(values, values.T)
        np.fill_diagonal(ortoMatrix, 0)
        mean = np.abs(ortoMatrix).sum() / (len(letters) * (len(letters) - 1))
        unique, counts = np.unique(np.abs(ortoMatrix.flatten()), return_counts=True)
        counter = {}
        for u, c in zip(unique, counts):
            if u == 0:
                continue
            counter[int(u)] = int(c) // 2
        return mean, counter
    
    @staticmethod
    def saveHopfieldOrtogonalityOutput(filename, df, counters):
        filepath = os.path.join(os.path.dirname(sys.argv[0]), "results", "hopfield", filename)
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w") as file:
            writer = csv.DictWriter(file, fieldnames=["group", "mean", "counter"])
            writer.writeheader()
            for i, row in df.iterrows():
                writer.writerow({"group": row["Group"], "mean": row["Mean"], "counter": counters[row["Group"]]})
                
    @staticmethod
    def saveHopfieldLetterPerEpoch(filename, letterPerEpoch):
        filepath = os.path.join(os.path.dirname(sys.argv[0]), "results", "hopfield", filename)
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w") as file:
            writer = csv.writer(file)
            for epoch in letterPerEpoch:
                writer.writerow(epoch)
