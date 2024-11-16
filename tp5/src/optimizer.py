from abc import ABC, abstractmethod
import numpy as np


class OptimizerFunction(ABC):
    def __init__(self, options=None):
        self.options = options

    @abstractmethod
    def __call__(self, gradient, **kwargs):
        pass

    @staticmethod
    def getFunction(name):
        name = name.lower()
        if name == "gradient_descent":
            return GradientDescentOptimizer
        elif name == "momentum":
            return MomentumOptimizer
        elif name == "adam":
            return AdamOptimizer
        else:
            raise ValueError("Invalid optimizer function")


class GradientDescentOptimizer(OptimizerFunction):
    def __call__(self, gradient, **kwargs):
        return self.options["rate"] * gradient


class MomentumOptimizer(OptimizerFunction):
    def __call__(self, gradient, **kwargs):
        previousDeltaW = kwargs["previousDeltaW"]
        if previousDeltaW is None:
            raise ValueError("Momentum optimizer requires previous deltaW")

        return self.options["rate"] * gradient + self.options["alpha"] * previousDeltaW


class AdamOptimizer(OptimizerFunction):
    EPSILON = 1e-7

    def __init__(self, options):
        super().__init__(options)
        self.m = None
        self.v = None
        self.t = 0

    def __call__(self, gradient, **kwargs):
        if self.m is None:
            self.m = np.zeros(np.shape(gradient))
            self.v = np.zeros(np.shape(gradient))

        self.t += 1
        self.m = self.options["beta1"] * self.m + (1 - self.options["beta1"]) * gradient
        self.v = self.options["beta2"] * self.v + (1 - self.options["beta2"]) * np.power(gradient, 2)

        mHat = self.m / (1 - np.power(self.options["beta1"], self.t))
        vHat = self.v / (1 - np.power(self.options["beta2"], self.t))

        return self.options["rate"] * mHat / (np.sqrt(vHat) + self.EPSILON)
