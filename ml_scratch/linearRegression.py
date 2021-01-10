import numpy as np


class linearRegression:
    def __init__(self):
        self.w = None

    def predict(self, X_test):
        N, M = X_test.shape
        Xc = self._add_bias_col(X_test)
        y = Xc.dot(self.w).squeeze()
        return y

    def fit(self, X, y, method='analytic', iterations=100, nu=0.001):
        if method == 'analytic':
            self._find_w_analytic(X, y)
        else:
            self._find_w_numerical(X, y, iterations=iterations, nu=nu)

    def _find_w_numerical(self, X, y, iterations=100, nu=0.001):
        N, M = X.shape
        Xc = self._add_bias_col(X)
        y = y[:, None]
        self.w = np.zeros((M+1, 1))
        for i in range(iterations):
            grad = Xc.T.dot(Xc.dot(self.w) - y)
            self.w = self.w - nu * grad
            if i % 10 == 0:
                cost = 0.5 * np.mean((Xc.dot(self.w) - y)**2)
                print(f'iteration: {i}, cost: {cost}')

    def _find_w_analytic(self, X, y):
        N, M = X.shape
        Xc = self._add_bias_col(X)
        self.w = np.linalg.inv(Xc.T.dot(Xc)).dot(Xc.T).dot(y)

    def _add_bias_col(self, X):
        N, _ = X.shape
        return np.concatenate([np.ones((N, 1)), X], axis=1)
