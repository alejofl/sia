import numpy as np
import os
import sys
import json


class Constants:
    _instance = None

    def __new__(cls, **kwargs):
        if cls._instance is None:
            cls._instance = super(Constants, cls).__new__(cls)
            cls._instance._initialize(**kwargs)
        return cls._instance

    def _initialize(self, **kwargs):
        self.seed = kwargs.get("seed", None)
        self.random = np.random.default_rng(self.seed)
        with open(os.path.join(os.path.dirname(sys.argv[0]), "resources", "countries.json"), 'r') as countries:
            self.countryAbbreviations = json.load(countries)

    @staticmethod
    def getInstance():
        if Constants._instance is None:
            raise ValueError("Singleton class hasn't been instantiated")
        return Constants._instance
