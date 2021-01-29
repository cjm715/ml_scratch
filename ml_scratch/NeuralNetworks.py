import numpy as np


class NeuralNetwork:
    def __init__(self, num_layers = 2, input_size = 64, num_nodes= [30, 10]):
        self.num_layers = num_layers
        self.input_size = input_size
        self.num_nodes = num_nodes

        self.W = [np.random.randn(num_nodes[0], input_size)]
        self.b = [np.random.randn(num_nodes[0], 1)]
        for i in range(1, num_layers):
            self.W.append(np.random.randn(num_nodes[i], num_nodes[i-1]))
            self.b.append(np.random.randn(num_nodes[i], 1))
        self.a = [np.zeros((num_nodes[i], 1)) for i in range(num_layers)]
        self.z = [np.zeros((num_nodes[i], 1)) for i in range(num_layers)]

    def fit(self, X, y):
        pass

    def predict(self, X):
        pass

    def _forward(self, X):
        for i in range(self.num_layers):
            if i == 0:
                self.z[i] = self.W[i].dot(X.T) + self.b[i]
            else:
                self.z[i] = self.W[i].dot(self.a[i-1]) + self.b[i]

            if i < (self.num_layers - 1):
                self.a[i] = ReLU(self.z[i])
            else: # layer i is the final layer
                self.a[i] = softmax(self.z[i])

        return self.a[-1]

    def _backward(self):
        pass

def ReLU(z):
    z[z <= 0] = 0
    return z

def softmax(z):
    a = np.exp(z)
    a = a / np.sum(a, axis = 0)
    return a

from sklearn.datasets import load_digits
import matplotlib.pyplot as plt

if __name__ == "__main__":


    nn = NeuralNetwork()

    # print(len(nn.W))
    # print(nn.W[0].shape)
    # print(nn.W[1].shape)
    # print(len(nn.b))
    # print(nn.b[0].shape)
    # print(nn.b[1].shape)
    # print(len(nn.a))
    # print(nn.a[0].shape)
    # print(nn.a[1].shape)
    # print(len(nn.z))
    # print(nn.z[0].shape)
    # print(nn.z[1].shape)

    X,y = load_digits(return_X_y = True)

    yp = nn._forward(X)
    print(yp.shape)
    print(np.sum(yp, axis = 0))
