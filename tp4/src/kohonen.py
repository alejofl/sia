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

            winnerNeuron = np.array(self.test(x))

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

    def test(self, x):
        winnerI, winnerJ = None, None
        minDistance = np.inf
        for i in range(self.matrixSquareSize):
            for j in range(self.matrixSquareSize):
                distance = self.distanceCalculator(x[1:], self.matrix[i][j])
                if distance < minDistance:
                    winnerI, winnerJ = i, j
                    minDistance = distance
        return winnerI, winnerJ
    
    def averageNeighborDistances(self):
        avg_distance_matrix = np.zeros((self.matrixSquareSize, self.matrixSquareSize))
        for i in range(self.matrixSquareSize):
            for j in range(self.matrixSquareSize):
                distances = []
                for di in [-1, 0, 1]:
                    for dj in [-1, 0, 1]:
                        ni, nj = i + di, j + dj
                        if 0 <= ni < self.matrixSquareSize and 0 <= nj < self.matrixSquareSize and (di != 0 or dj != 0):
                            distance = self.distanceCalculator(self.matrix[i][j], self.matrix[ni][nj])
                            distances.append(distance)
                avg_distance_matrix[i, j] = np.mean(distances) if distances else 0
        return avg_distance_matrix
