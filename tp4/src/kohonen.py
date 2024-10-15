import numpy as np
from .constants import Constants
from .distance import EuclideanDistance


class Kohonen:
    def __init__(self, **kwargs):
        self.trainingSet = kwargs.get('trainingSet', None)
        self.matrixSquareSize = kwargs.get('matrixSquareSize', None)
        self.maxEpochs = kwargs.get('maxEpochs', None)
        self.initialLearningRate = kwargs.get('initialLearningRate', 0.1)
        self.initialRadius = kwargs.get('initialRadius', 1)
        self.changeLearningRate = kwargs.get('changeLearningRate', True)
        self.changeRadius = kwargs.get('changeRadius', False)
        self.distanceCalculator = kwargs.get('distanceCalculator', EuclideanDistance())
        self.matrixPerEpoch = []

        if self.trainingSet is None or self.matrixSquareSize is None or self.maxEpochs is None:
            raise ValueError()

        self.initializeWeights()

    def initializeWeights(self):
        bucket = []
        for i in range(len(self.trainingSet)):
            for j in range(1, len(self.trainingSet[i])):
                bucket.append(self.trainingSet[i][j])

        self.matrix = Constants.getInstance().random.choice(bucket, size=(self.matrixSquareSize, self.matrixSquareSize, len(self.trainingSet[0]) - 1))

    def train(self):
        for epoch in range(self.maxEpochs):
            self.matrixPerEpoch.append(np.copy(self.matrix))
            x = Constants.getInstance().random.choice(self.trainingSet)

            winnerI, winnerJ = None, None
            minDistance = np.inf
            for i in range(self.matrixSquareSize):
                for j in range(self.matrixSquareSize):
                    distance = self.distanceCalculator(x[1:], self.matrix[i][j])
                    if distance < minDistance:
                        winnerI, winnerJ = i, j
                        minDistance = distance
            winnerNeuron = np.array([winnerI, winnerJ])

            for i in range(self.matrixSquareSize):
                for j in range(self.matrixSquareSize):
                    neuron = np.array([i, j])
                    if np.linalg.norm(winnerNeuron - neuron) <= self.radius(epoch):
                        self.matrix[i][j] += self.learningRate(epoch) * (x[1:] - self.matrix[i][j])

    def radius(self, epoch):
        if not self.changeRadius:
            return self.initialRadius
        return np.max([self.initialRadius * (1 - epoch / self.maxEpochs), 1])

    def learningRate(self, epoch):
        if not self.changeLearningRate:
            return self.initialLearningRate
        return 1 / (epoch + 1)
