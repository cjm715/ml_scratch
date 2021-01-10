import numpy as np


class GDA:
    ''' Gaussian Discriminant Analysis

    This model is for binary classification with the assumption that the data is
    modeled by multivariate guassians in feature space for each class. See
    http://cs229.stanford.edu/notes2020spring/cs229-notes2.pdf for more
    details.

    Attributes:

        params (dict): dictionary of model parameters.

    '''

    def __init__(self):
        pass

    def fit(self, X, y):
        '''
        Fits model to training features X and labels y.

        Args:
            X (numpy.array): matrix with n examples (each row) and m features
                (each column).
            y (np.array): array of 0 or 1 indicating the class of this example.
                Array has size (n, 1) where n is the number of examples.
        '''
        pass

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
        pass
