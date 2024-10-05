import numpy as np

class Metrics:
    def __init__(self, expectedOutputs, predictedOutputs, error):
        self.TP = 0
        self.FN = 0
        self.FP = 0
        self.TN = 0
        self.error = error
        # TODO: Create confusion matrix
        # for expected, predicted in zip(expectedOutputs, predictedOutputs):
        #     if expected-predicted==0:
        #         self.TP += 1
        #     elif expected == 1 and predicted == 0:
        #         self.FN += 1
        #     elif expected == 0 and predicted == 1:
        #         self.FP += 1
        #     elif expected == 0 and predicted == 0:
        #         self.TN += 1

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
        