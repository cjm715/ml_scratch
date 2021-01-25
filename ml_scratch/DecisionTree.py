import numpy as np
from collections import namedtuple

Split = namedtuple('Split','feature_idx threshold split_entropy')


class DecisionTree:
    '''
    Binary decision tree classifier.

    Attributes:
        root (dict): the root node of tree (and each node in tree) is represented as a
            dictionary with keys:
            - 'id' (int): node id
            - 'num_samples' (int): number of samples
            - 'frac_positive' (float): fraction of positive labels in train set.
            - 'entropy' (float): entropy of node,
            - 'split' (Split [namedtuple], exists for non-leaf nodes):
                split information such as
                feature, threshold, and split entropy
            - 'left' (dict, exists for non-leaf nodes): left node representing
                data less than or equal to threshold in split. This has the same
                dictionary structure as defined here for root node.
            - 'right' (dict,  exists for non-leaf nodes): right node representing
                data greater than threshold in split. This has the same dictionary
                structure as defined here for root node.

    Args:
        max_depth (int): maximum depth of tree to fit to data. This is a
            regularization parameter.
        max_num_nodes (int): maximum number of nodes in tree. This is a
            regularization parameter.

    '''

    def __init__(self, max_depth=5, max_num_nodes = 15):
        self.root = {}
        self.max_depth = max_depth
        self.max_num_nodes = max_num_nodes
        self._id_generator = _numberGenerator()

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
        level = 0
        node = {}
        self.root = self._build_tree(X, y, node, level)

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
        # add additional column with original row id since data will
        # be split up in recursion algorithm within _calc_prob_from_tree.
        # row id will travel along with data during traversal and then sorted
        # at end by row id.
        idx_col =  np.arange(X.shape[0], dtype = np.float).reshape(X.shape[0],1)
        X = np.append(X, idx_col, 1)

        p = _calc_prob_from_tree(X, self.root)

        # sort by row id and remove row id column in output
        p = p[np.argsort(p[:, -1])]
        p = p[:, 0]

        return p

    def _build_tree(self, X, y, node, level):
        p = _calc_fraction_positive(y)
        entropy = _calc_entropy(y)
        node['id'] = next(self._id_generator)
        node['num_samples'] = len(y)
        node['frac_positive'] = p
        node['entropy'] = entropy
        if p == 0 or p == 1:
            return node

        # add 1 to count up to parent and add 2 to consider children if split further
        num_nodes_with_potential_split = node['id'] + 1 + 2

        if  num_nodes_with_potential_split >= self.max_num_nodes:
            return node

        level = level + 1
        if level > self.max_depth:
            return node

        split = _find_optimal_split(X, y)
        Xl, yl, Xr, yr = _split_data(split.feature_idx, split.threshold, X, y)
        node['split'] = split
        node['left'] = self._build_tree(Xl, yl, {}, level)
        node['right'] = self._build_tree(Xr, yr, {}, level)

        return node

def _split_features(feature_idx, threshold, X):
    # right for data with feature greater than threshold
    mask_right = X[:, feature_idx] > threshold
    mask_left = np.logical_not(mask_right)

    X_left = X[mask_left, :]
    X_right = X[mask_right, :]

    return X_left, X_right

def _calc_prob_from_tree(X, node):
    if 'split' in node:
        # Reached a split in decision tree. ask children for probabilities
        # and concatenate
        split = node['split']
        Xl, Xr = _split_features(split.feature_idx, split.threshold, X)
        pl = _calc_prob_from_tree(Xl, node['left'])
        pr = _calc_prob_from_tree(Xr, node['right'])
        p = np.append(pl, pr, 0)
        return p
    else:
        # leaf node; predict fraction positive from training set as probability
        # of positive class.
        p = np.ones((X.shape[0],1), dtype= np.float)*node['frac_positive']
        idx_col = X[:, -1].reshape(X.shape[0],1)
        p = np.append(p, idx_col, 1)

        return p

def _calc_fraction_positive(y):
    if len(y) == 0:
        return 0
    return sum(y == 1)/len(y)

def _calc_entropy(y):
    p = _calc_fraction_positive(y)
    if p == 0 or p == 1:
        return 0.
    q = 1-p
    entropy = - p*np.log(p) - q*np.log(q)
    return entropy

def _calc_split_entropy(feature_idx, threshold, X, y):

    _, y_left, _, y_right = _split_data(feature_idx, threshold, X, y)

    H_left = _calc_entropy(y_left)
    H_right = _calc_entropy(y_right)

    p_left = len(y_left)/len(y)
    p_right = len(y_right)/len(y)

    H = p_left*H_left + p_right*H_right

    return H



def _split_data(feature_idx, threshold, X, y):
    # right for data with feature greater than threshold
    mask_right = X[:, feature_idx] > threshold
    mask_left = np.logical_not(mask_right)

    X_left = X[mask_left, :]
    y_left = y[mask_left]

    X_right = X[mask_right, :]
    y_right = y[mask_right]

    return X_left, y_left, X_right, y_right


def _find_optimal_split(X, y):
    # TODO: This function could be optimized by using information from split with
    # previous threshold.
    cand_split_list = []
    for feature_idx in range(X.shape[1]):
        x_values = np.sort(X[:, feature_idx])
        # midpoints of all x_values
        threshold_list = (x_values[:-1] + x_values[1:])/2.
        for threshold in threshold_list:
            split_entropy = _calc_split_entropy(feature_idx, threshold, X, y)
            cand_split = Split(feature_idx = feature_idx,
                               threshold = threshold,
                               split_entropy = split_entropy)
            cand_split_list.append(cand_split)
    optimal_split = min(cand_split_list, key=lambda split: split.split_entropy)
    return optimal_split

def _numberGenerator():
    number = 0
    while True:
        yield number
        number += 1
