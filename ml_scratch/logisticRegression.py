import numpy as np


def add_bias_col(X):
    N, _ = X.shape
    return np.concatenate([np.ones((N, 1)), X], axis=1)


def sigmoid(z):
    return 1/(1+np.exp(-z))


class logisticRegression:
    def __init__(self):
        pass

    def fit(self, X, y, iterations=100, nu=0.001):
        N, M = X.shape
        Xc = add_bias_col(X)
        y = y[:, None]
        self.w = np.zeros((M + 1, 1))
        for i in range(iterations):
            grad = Xc.T.dot(y - self._hypothesis(Xc))
            self.w = self.w + nu * grad
            if i % 10 == 0:
                cost = y.T.dot(np.log(self._hypothesis(Xc)))
                cost += (1 - y).T.dot(np.log(1 - self._hypothesis(Xc)))
                cost = cost.squeeze()
                print(f'iteration: {i}, cost: {cost}')

    def fit_newtons_method(self, X, y, iterations=100, nu=0.001):
        N, M = X.shape
        Xc = add_bias_col(X)
        y = y[:, None]
        # self.w = np.random.uniform(size=(M + 1, 1))
        self.w = np.zeros((M + 1, 1))
        for i in range(iterations):
            yhat = self._hypothesis(Xc).squeeze()
            deriv_sigmoid = np.diag(yhat*(1 - yhat))
            H = - Xc.T.dot(deriv_sigmoid.dot(Xc))
            grad = Xc.T.dot(y - self._hypothesis(Xc))
            delw = np.linalg.solve(H, -grad)
            self.w = self.w + nu*delw
            cost = y.T.dot(np.log(self._hypothesis(Xc)))
            cost += (1 - y).T.dot(np.log(1 - self._hypothesis(Xc)))
            cost = cost.squeeze()
            print(f'iteration: {i}, cost: {cost}')

    def predict(self, X_test):
        N, M = X_test.shape
        Xc = add_bias_col(X_test)
        y = self._hypothesis(Xc).squeeze()
        return y

    def _hypothesis(self, Xc):
        return sigmoid(Xc.dot(self.w))
