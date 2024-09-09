import numpy as np


class RandomGenerator:
    _instance = None

    def __new__(cls, seed):
        if cls._instance is None:
            cls._instance = super(RandomGenerator, cls).__new__(cls)
            cls._instance._initialize(seed)
        return cls._instance

    def _initialize(self, seed):
        self.seed = seed
        self.generator = np.random.default_rng(seed)

    @staticmethod
    def getInstance():
        if RandomGenerator._instance is None:
            raise ValueError("Random Generator hasn't been instantiated")
        return RandomGenerator._instance
