import numpy as np


class Hopfield:
    def __init__(self, **kwargs):
        self.patterns = kwargs.get('patterns', None)
        self.maxEpochs = kwargs.get('maxEpochs', None)
        if self.patterns is None or self.maxEpochs is None:
            raise ValueError()
        self.patternLength = len(self.patterns[0])
        k = np.array(self.patterns)
        self.weights = 1 / self.patternLength * np.matmul(k.T, k)
        np.fill_diagonal(self.weights, 0)

    def test(self, pattern):
        patternPerEpoch = []
        pattern = np.array(pattern)
        for _ in range(self.maxEpochs):
            patternPerEpoch.append(np.copy(pattern))
            pattern = np.sign(np.matmul(self.weights, pattern.T))
            if np.array_equal(pattern, patternPerEpoch[-1]):
                break
        return patternPerEpoch
