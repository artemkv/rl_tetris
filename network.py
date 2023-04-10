import numpy as np
import math

# learning rate
alpha = 0.001

# hidden neurons
NH = 100

# output neurons
NO = 5  # 5 actions


class NeuralNetwork:
    def __init__(self, D):
        # hidden weights, Xavier initialization
        self.WH = np.random.randn(NH, D) * math.sqrt(2.0 / D)

        # Output layer weights, Xavier initialization
        self.WO = np.random.randn(NO, NH) * math.sqrt(2.0 / NH)

        self.X = None
        self.HO = None
        self.OO = None
        self.Y = None

    def forward(self, X):
        # remember input
        self.X = np.copy(X)

        # hidden layer calculations
        self.HO = X.dot(self.WH.T)
        HO_sigmoid = self.leaky_relu(self.HO)

        # output layer calculations
        self.OO = HO_sigmoid.dot(self.WO.T)
        self.Y = self.relu(self.OO)

        return np.copy(self.Y)

    def leaky_relu(self, x):
        return np.where(x > 0, x, x * 0.01)

    def relu(self, x):
        return np.where(x > 0, x, 0)

    def backprop(self, targets):
        x = 1
