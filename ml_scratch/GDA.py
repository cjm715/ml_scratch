import numpy as np


class GDA:
    ''' (Binary) Gaussian Discriminant Analysis

    This model is for binary classification with the assumption that the data is
    modeled by multivariate guassians in feature space for each class. See
    http://cs229.stanford.edu/notes2020spring/cs229-notes2.pdf for more
    details.

    Attributes:
        params (dict): dictionary of model parameters.

    '''

    def __init__(self):
        self.params = {}

    def fit(self, X, y):
        '''
        Fits model to training features X and labels y.

        Args:
            X (numpy.array): matrix with n examples (each row) and m features
                (each column).
            y (np.array): array of 0 or 1 indicating the class of this example.
                Array has size (n, 1) where n is the number of examples.
        '''

        self.params['mu0'] = np.mean(X[y == 0, :], axis=0)
        self.params['mu1'] = np.mean(X[y == 1, :], axis=0)
        self.params['sigma0'] = np.cov(X[y == 0, :].T)
        self.params['sigma1'] = np.cov(X[y == 1, :].T)
        self.params['phi'] = np.sum(y == 1)/len(y)

    def predict(self, X_test):
        '''
        Predict class for input data matrix X_test

        Args:
            X_test (numpy.array): matrix with n examples (each row) and m
                features (each column).

        Returns:
            (np.array): probability of class 1 with size (n, ) where n is the
                number of examples.
        '''

        posterior = np.zeros(X_test.shape)
        for target_class in range(2):
            mu = self.params[f'mu{target_class}']
            sigma = self.params[f'sigma{target_class}']
            phi = self.params['phi']
            likelihood = gaussian_pdf(X_test, mu, sigma)
            prior = phi**(target_class)*(1-phi)**(1-target_class)
            posterior[:, target_class] = likelihood * prior

        prob = posterior / np.sum(posterior, axis=1)[:, None]
        return prob[:, 1]


def gaussian_pdf(X, mu, sigma):
    ''' Returns guassian (with mean mu and covariance sigma) pdf values for a
    collection of points.
    '''

    det = np.linalg.det(sigma)
    sigma_inv = np.linalg.inv(sigma)
    exponent = - 0.5*np.sum(((X - mu).T*sigma_inv.dot((X - mu).T)), axis=0)
    prob_density = 1./(2. * np.pi*np.sqrt(det)) * np.exp(exponent)
    return prob_density
