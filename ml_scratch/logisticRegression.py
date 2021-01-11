import numpy as np


class logisticRegression:
    ''' Logistic Regression model

    This model is for binary classification. For more details,
    see: https://youtu.be/het9HFqo1TQ?list=PLoROMvodv4rMiGQp3WXShtMGgzqpfVfbU&t=2779

    Attributes:
        w (np.array): array of weights in logistic regression where w[0] is the
            intercept term.

    '''

    def __init__(self):
        self.w = None

    def fit(self, X, y, iterations=None, nu=None, method='gd'):
        '''
        Fits model to training features X and labels y.

        Args:
            X (numpy.array): matrix with n examples (each row) and m features
                (each column).
            y (np.array): array of 0 or 1 indicating the class of this example.
                Array has size (n, 1) where n is the number of examples.
            method (string): choose solving method with method == 'gd' for
                gradient descent or method == 'newton' for Newton's method.
            iterations (int, optional): number of iterations for solver
            nu (float, optional): step size for solver.
        '''
        if method == 'gd':
            if iterations is None:
                iterations = 100
            if nu is None:
                nu = 0.01
            self._fit_gd(X, y, iterations=iterations, nu=nu)
        else:  # method == 'newton'
            if iterations is None:
                iterations = 10
            if nu is None:
                nu = 1
            self._fit_newton(X, y, iterations=iterations, nu=nu)

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
        Xc = add_bias_col(X_test)
        y = self._hypothesis(Xc).squeeze()
        return y

    def _fit_gd(self, X, y, iterations=100, nu=0.001):
        ''' fit with gradient descent '''

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

    def _fit_newton(self, X, y, iterations=100, nu=1):
        ''' fit with Newton's method '''

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

    def _hypothesis(self, Xc):
        return sigmoid(Xc.dot(self.w))


def add_bias_col(X):
    N, _ = X.shape
    return np.concatenate([np.ones((N, 1)), X], axis=1)


def sigmoid(z):
    return 1/(1+np.exp(-z))
