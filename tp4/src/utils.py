import csv
import os
import sys
import numpy as np
import pandas as pd


class Utils:
    EUROPE_DATASET_PATH = os.path.join(os.path.dirname(sys.argv[0]), "resources", "europe.csv")

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

