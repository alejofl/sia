import numpy as np
from .constants import Constants


class Oja:
    def __init__(self, **kwargs):
        self.data = kwargs.get('data', None)
        self.maxEpochs = kwargs.get('maxEpochs', None)
        self.initialLearningRate = kwargs.get('initialLearningRate', 0.1)
        self.changeLearningRate = kwargs.get('changeLearningRate', True)
        if self.data is None or self.maxEpochs is None:
            raise ValueError()
        self.weights = Constants.getInstance().random.random(len(self.data[0]))

    def run(self):
        for epoch in range(self.maxEpochs):
            for input in self.data:
                output = np.dot(input, self.weights)
                deltaW = self.learningRate(epoch) * output * (input - output * self.weights)
                self.weights = np.add(self.weights, deltaW)
        return self.weights

    def learningRate(self, epoch):
        if not self.changeLearningRate:
            return self.initialLearningRate
        return self.initialLearningRate / (epoch + 1)
