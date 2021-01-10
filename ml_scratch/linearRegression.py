import numpy as np


class linearRegression:
    ''' Linear Regression Model

    This is a classic regression model which assumes the output variable is
    linearly dependent on the input variables. For more details,
    see: https://youtu.be/4b4MUYve_U8?list=PLoROMvodv4rMiGQp3WXShtMGgzqpfVfbU&t=284

    Attributes:
        w (np.array): array of weights in linear regression where w[0] is the
            intercept term.

    '''

    def __init__(self):
        self.w = None

    def fit(self, X, y, method='analytic', iterations=100, nu=0.001):
        '''
        Fits model to training features X and labels y.

        Args:
            X (numpy.array): matrix with n examples (each row) and m features
                (each column).
            y (np.array): array of 0 or 1 indicating the class of this example.
                Array has size (n, 1) where n is the number of examples.
            method (string, optional): can be either 'analytic' (uses normal
                equation to solve) or 'gd' (uses gradient descent to solve)
            iterations (int, optional): number of iterations in gradient
                descent. only used when method == 'gd'.
            nu (float, optional): step size for gradient descent. Only
                effective if method == 'gd'

        '''
        if method == 'analytic':
            self._find_w_analytic(X, y)
        else:  # method == 'gd'
            self._find_w_grad_descent(X, y, iterations=iterations, nu=nu)

    def predict(self, X_test):
        '''
        Predict class for input data matrix X_test

        Args:
            X_test (numpy.array): matrix with n examples (each row) and m
                features (each column).

        Returns:
            (np.array): probability of class 1 with size (n, 1) where n is the
                number of examples.
        '''
        N, M = X_test.shape
        Xc = self._add_bias_col(X_test)
        y = Xc.dot(self.w).squeeze()
        return y

    def _find_w_grad_descent(self, X, y, iterations=100, nu=0.001):
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
