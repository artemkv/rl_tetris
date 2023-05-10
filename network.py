import numpy as np
import math

# learning rate
alpha = 0.0001

# hidden neurons
NH = 100

# output neurons
NO = 5  # 5 actions


class NeuralNetwork:
    def __init__(self, INPUT_SIZE):
        self.INPUT_SIZE = INPUT_SIZE

        # hidden weights, Xavier initialization
        self.WH = np.random.randn(INPUT_SIZE, NH) * math.sqrt(2.0 / INPUT_SIZE)

        # Output layer weights, Xavier initialization
        self.WO = np.random.randn(NH, NO) * math.sqrt(2.0 / NH)

        self.X = None
        self.HO = None
        self.OO = None
        self.Y = None

    def copy(self):
        nn = NeuralNetwork(self.INPUT_SIZE)
        nn.WH = np.copy(self.WH)
        nn.WO = np.copy(self.WO)
        return nn

    def save(self, filename):
        np.savetxt(f'{filename}_hid.csv', self.WH, delimiter=',')
        np.savetxt(f'{filename}_out.csv', self.WO, delimiter=',')

    def load(self, filename):
        self.WH = np.genfromtxt(f'{filename}_hid.csv', delimiter=',')
        self.WO = np.genfromtxt(f'{filename}_out.csv', delimiter=',')

    def forward(self, X):
        # save input
        self.X = X

        # hidden layer calculations
        self.HO = X.dot(self.WH)
        HO_relu = self.leaky_relu(self.HO)

        # output layer calculations
        self.OO = HO_relu.dot(self.WO)
        self.Y = self.leaky_relu(self.OO)

        return self.Y

    def leaky_relu(self, x):
        return np.where(x > 0, x, x * 0.01)

    def relu(self, x):
        return np.where(x > 0, x, 0)

    def relu_derivative(self, x):
        return np.where(x > 0, 1, 0)

    def leaky_relu_derivative(self, x):
        return np.where(x > 0, 1, 0.01)

    def backprop(self, targets):
        # gradient on the error function
        err_grad = self.Y - targets

        # gradient on the output layer
        grad_o = err_grad * self.leaky_relu_derivative(self.OO)

        # gradient going to the weights of the output layer
        grad_wo = self.HO.reshape((1, NH)).T.dot(grad_o.reshape((1, NO)))

        # gradient going to the neurons of the hidden layer
        grad_h = grad_o.dot(self.WO.T)

        # gradient on the hidden layer
        grad_h1 = grad_h * self.leaky_relu_derivative(self.HO)

        # gradient going to the weights of the hidden layer
        grad_wh = self.X.reshape((1, self.INPUT_SIZE)).T.dot(
            grad_h1.reshape((1, NH)))

        # update weights
        self.WO = self.WO - alpha * grad_wo
        self.WH = self.WH - alpha * grad_wh
