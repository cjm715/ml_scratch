import numpy as np

class RandomForest:
    '''
    Random Forest classifier.

    '''

    def __init__(self):
        pass

    def fit(self, X, y):
        '''
        Fits model to training features X and labels y.

        Args:
            X (numpy.array): matrix with n examples (each row) and m features
                (each column).
            y (numpy.array): array of 0 (-1 works as well for negative class) or 1
                indicating the class of this example. Array has size (n, 1)
                where n is the number of examples.
        '''
        pass

    def predict(self, X):
        '''
        Predict class for input data matrix X_test

        Args:
            X (numpy.array): matrix with n examples (each row) and m
                features (each column).

        Returns:
            (np.array): probability of class 1 with size (n, 1) where n is the
                number of examples.
        '''
        pass
