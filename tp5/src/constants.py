import numpy as np


class Constants:
    _instance = None

    def __new__(cls, **kwargs):
        if cls._instance is None:
            cls._instance = super(Constants, cls).__new__(cls)
            cls._instance._initialize(**kwargs)
        return cls._instance

    def _initialize(self, **kwargs):
        self.epsilon = kwargs.get("epsilon", 3e-2)
        self.maxEpochs = kwargs.get("maxEpochs", 1000)
        self.batchSize = kwargs.get("batchSize", 1) # 1: online; -1: batch; 1 < x < len(inputs): mini-batch
        self.seed = kwargs.get("seed", None)
        self.random = np.random.default_rng(self.seed)
        self.bias = 1

    @staticmethod
    def getInstance():
        if Constants._instance is None:
            raise ValueError("Singleton class hasn't been instantiated")
        return Constants._instance
