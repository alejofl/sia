import numpy as np
from sklearn.metrics import confusion_matrix

class Metrics:
    def __init__(self, expectedOutputs, predictedOutputs):
        self.confusionMatrix = confusion_matrix(expectedOutputs, predictedOutputs).ravel()
        self.TN, self.FP, self.FN, self.TP = self.confusionMatrix

    def accuracy(self):
        total = self.TP + self.TN + self.FP + self.FN
        return (self.TP + self.TN) / total if total!=0 else 0

    def precision(self):
        denominator = self.TP + self.FP
        return self.TP / (self.TP + self.FP) if denominator!=0 else 0

    def recall(self):
        denominator = self.TP + self.FN
        return self.TP / (self.TP + self.FN) if denominator!=0 else 0

    def f1Score(self):
        precision = self.precision()
        recall = self.recall()
        precisionAndRecall = precision + recall
        return 2 * precision * recall / (precisionAndRecall) if precisionAndRecall!=0 else 0
    
    def TPRate(self):
        total = self.TP + self.FN
        return self.TP / total if total!=0 else 0
    
    def FPRate(self):
        total = self.FP + self.TN
        return self.FP / (total) if total!=0 else 0
    
    def displayAll(self):
        print("---Metrics---")
        print("Accuracy: ", self.accuracy())
        print("Precision: ", self.precision())
        print("Recall", self.recall())
        print("F1-Score", self.f1Score())
        print("TP Rate", self.TPRate())
        print("FP Rate", self.FPRate())
        print("----------")
        return