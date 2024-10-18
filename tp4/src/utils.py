import os
import sys
import numpy as np
import pandas as pd


class Utils:
    EUROPE_DATASET_PATH = os.path.join(os.path.dirname(sys.argv[0]), "resources", "europe.csv")

    @staticmethod
    def getEuropeDataset():
        df = pd.read_csv(Utils.EUROPE_DATASET_PATH)
        df = Utils.zScoreData(df)
        countries = df["Country"].to_numpy()
        data = df.reset_index().drop(columns=["Country"]).to_numpy()
        return countries, data

    @staticmethod
    def zScoreData(df):  # TODO: Check
        def zScore(col):
            mean = col.mean()
            std = col.std()
            return col.apply(lambda x: (x - mean) / std)

        return df.apply(lambda col: zScore(col) if col.name != "Country" else col)
