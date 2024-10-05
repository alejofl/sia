import numpy as np
from sklearn.metrics import confusion_matrix

class Metrics:
    def __init__(self, expectedOutputs, predictedOutputs, binary=True):
        self.TP = 0
        self.FN = 0
        self.FP = 0
        self.TN = 0
        self.confusionMatrix = confusion_matrix(expectedOutputs, predictedOutputs).ravel() if binary==True else confusion_matrix(expectedOutputs, predictedOutputs)
        self.TN, self.FP, self.FN, self.TP = self.confusionMatrix

    def accuracy(self):
        return (self.TP + self.TN) / (self.TP + self.TN + self.FP + self.FN)

    def precision(self):
        return self.TP / (self.TP + self.FP)

    def recall(self):
        return self.TP / (self.TP + self.FN)

    def f1Score(self):
        precision = self.precision()
        recall = self.recall()
        return 2 * precision * recall / (precision + recall)
    
    def TPRate(self):
        return self.TP / (self.TP + self.FN)
    
    def FPRate(self):
        return self.FP / (self.FP + self.TN)
        