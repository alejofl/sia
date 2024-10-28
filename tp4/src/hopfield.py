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
            pattern = self.sign(pattern, np.matmul(self.weights, pattern.T))
            if np.array_equal(pattern, patternPerEpoch[-1]):
                break

        if not self.patternExists(patternPerEpoch[-1]):
            print("Did not converge")
        return patternPerEpoch
    
    def sign(self, previous, new):
        result = np.zeros(previous.shape)
        for i in range(len(new)):
                result[i] = np.sign(new[i])
                if result[i] == 0:
                    result[i] = previous[i]
        return result

    def patternExists(self, pattern):
        for p in self.patterns:
            if np.array_equal(p, pattern):
                return True
        return False
