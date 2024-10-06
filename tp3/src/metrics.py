import numpy as np
from sklearn.metrics import *

class Metrics:
    def __init__(self, expected, predicted):
        self.expected = expected
        self.predicted = predicted

    def accuracy(self):
        return accuracy_score(self.expected, self.predicted)

    def precision(self):
        return precision_score(y_true=self.expected, y_pred=self.predicted, average='macro', zero_division=0)

    def recall(self):
        return recall_score(y_true=self.expected, y_pred=self.predicted, average='macro', zero_division=0)

    def f1Score(self):
        return f1_score(y_true=self.expected, y_pred=self.predicted, average='macro', zero_division=0)
    
    def getAllMetrics(self):
        return [self.accuracy(), self.precision(), self.recall(), self.f1Score()]
    
    def displayAll(self):
        print("---Metrics---")
        print("Accuracy: ", self.accuracy())
        print("Precision: ", self.precision())
        print("Recall", self.recall())
        print("F1-Score", self.f1Score())
        print("-------------")
        return