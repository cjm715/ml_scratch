import numpy as np


class linearRegression:
    def __init__(self):
        pass

    def fit(self, X, y):
        N, M = X.shape
        Xc = np.concatenate([X, np.ones((N, 1))], axis=1)
        self.w = np.linalg.inv(Xc.T.dot(Xc)).dot(Xc.T).dot(y)

    def predict(self, X_test):
        N, M = X_test.shape
        Xc = np.concatenate([X_test, np.ones((N, 1))], axis=1)
        y = Xc.dot(self.w)
        return y
