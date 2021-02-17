import numpy as np


class NeuralNetwork:
    def __init__(self,
                num_layers = 2,
                input_size = 64,
                num_nodes= [30, 10],
                batch_size = 40,
                learning_rate = 0.1):

        self.num_layers = num_layers
        self.input_size = input_size
        self.num_nodes = num_nodes
        self.batch_size = batch_size
        self.learning_rate = learning_rate

        self.W = [np.random.randn(num_nodes[0], input_size)*0.01]
        self.b = [np.random.randn(num_nodes[0], 1)*0.01]
        for i in range(1, num_layers):
            self.W.append(np.random.randn(num_nodes[i], num_nodes[i-1])*0.01)
            self.b.append(np.random.randn(num_nodes[i], 1)*0.01)


    def fit(self, X, y, X_val=None, y_val=None, num_iterations = 10000):
        for itr in range(num_iterations):
            X_batch, y_batch = sample_batch(X, y, self.batch_size)

            a, z = self._forward(X_batch)
            dW, db = self._backward(X_batch, y_batch, a, z)
            for layer_idx in range(self.num_layers):
                self.W[layer_idx] -= self.learning_rate*dW[layer_idx]
                self.b[layer_idx] -= self.learning_rate*db[layer_idx]

            if itr % 100 == 0:
                y_hat = self.predict(X)
                if y_val is not None:
                    y_val_hat = self.predict(X_val)
                    print(itr,
                        "   Loss: ",
                        self._loss(y, y_hat),
                        "   Train Accuracy: ",
                        self._accuracy(y, y_hat),
                        "   Val Accuracy",
                        self._accuracy(y_val, y_val_hat))

                else:
                    print(itr,
                        "   Loss: ",
                        self._loss(y, y_hat),
                        "   Train Accuracy: ",
                        self._accuracy(y, y_hat))

    def predict(self, X):
        a, _ = self._forward(X)
        return a[-1].T

    def _loss(self, y, y_hat):
        return - np.mean(y*np.log(y_hat))

    def _accuracy(self, y, y_hat):
        is_correct = (np.argmax(y, axis = 1) == np.argmax(y_hat, axis = 1))
        #print(is_correct.shape)
        return sum(is_correct)/ len(is_correct)

    def _forward(self, X):
        num_instances = X.shape[0]
        a = [np.zeros((self.num_nodes[i], num_instances))
                  for i in range(self.num_layers)]
        z = [np.zeros((self.num_nodes[i], num_instances))
                  for i in range(self.num_layers)]
        for i in range(self.num_layers):
            if i == 0:
                z[i] = self.W[i].dot(X.T) + self.b[i]
            else:
                z[i] = self.W[i].dot(a[i-1]) + self.b[i]
            if i < (self.num_layers - 1):
                a[i] = ReLU(z[i])
            else: # layer i is the final layer
                a[i] = softmax(z[i])
        return a, z

    def _backward(self, X, y, a, z):
        num_instances = len(y)

        dW = [np.zeros(self.W[i].shape) for i in range(self.num_layers)]
        db = [np.zeros(self.b[i].shape) for i in range(self.num_layers)]

        # da = [np.zeros((self.num_nodes[i], num_instances))
        #           for i in range(num_layers)]
        dz = [np.zeros((self.num_nodes[i], num_instances))
                  for i in range(self.num_layers)]

        dz[-1] = a[-1] - y.T
        dW[-1] = (1/num_instances) *dz[-1].dot(a[-2].T)
        db[-1] = (1/num_instances) * np.sum(dz[-1], axis = 1, keepdims = True)

        for i in range(self.num_layers - 2, -1, -1):
            dz[i] =  derivReLU(z[i]) * self.W[i+1].T.dot(dz[i+1])
            if i == 0:
                dW[i] = (1/num_instances) * dz[i].dot(X)
            else:
                dW[i] = (1/num_instances) * dz[i].dot(a[i-1].T)
            db[i] = (1/num_instances) * np.sum(dz[i], axis = 1, keepdims = True)

        return dW, db


def sample_batch(X, y, batch_size):
    row_idx = np.random.choice(X.shape[0], batch_size, replace=False)
    X_batch = X[row_idx, :]
    y_batch = y[row_idx]
    return X_batch, y_batch

def derivReLU(z):
    deriv = np.zeros(z.shape)
    deriv[z > 0] = 1
    return deriv

def ReLU(z):
    z[z <= 0] = 0
    return z

def softmax(z):
    a = np.exp(z)
    a = a / np.sum(a, axis = 0)
    return a
