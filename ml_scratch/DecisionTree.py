import numpy as np
from collections import namedtuple

Split = namedtuple('Split','feature_idx threshold entropy')


class DecisionTree:
    def __init__(self, max_depth=5):
        self.root = {}
        self.max_depth = max_depth
        self.id_generator = numberGenerator()

    def fit(self, X, y):
        level = 0
        node = {}
        self.root = self._build_tree(X, y, node, level)

    def predict(self, X):
        pass

    def _build_tree(self, X, y, node, level):
        p = calc_p(y)
        node['id'] = next(self.id_generator)
        node['num_samples'] = len(y)
        node['frac_positive'] = p

        if p == 0 or p == 1:
            return node

        level = level + 1
        if level > self.max_depth:
            return node

        split = find_optimal_split(X, y)
        Xl, yl, Xr, yr = split_data(split.feature_idx, split.threshold, X, y)
        node['split'] = split
        node['left'] = self._build_tree(Xl, yl, {}, level)
        node['right'] = self._build_tree(Xr, yr, {}, level)

        return node

def calc_p(y):
    if len(y) == 0:
        return 0
    return sum(y == 1)/len(y)

def calc_gini(y):
    p = calc_p(y)
    G = 2*p*(1-p)
    return G

def calc_entropy(y):
    p = calc_p(y)
    if p == 0 or p == 1:
        return 0.
    q = 1-p
    entropy = - p*np.log(p) - q*np.log(q)
    return entropy

def split_data(feature_idx, threshold, X, y):
    # right for data with feature greater than threshold
    mask_right = X[:, feature_idx] > threshold
    mask_left = np.logical_not(mask_right)

    X_left = X[mask_left, :]
    y_left = y[mask_left]

    X_right = X[mask_right, :]
    y_right = y[mask_right]

    return X_left, y_left, X_right, y_right

def calc_split_entropy(feature_idx, threshold, X, y):

    _, y_left, _, y_right = split_data(feature_idx, threshold, X, y)

    H_left = calc_entropy(y_left)
    H_right = calc_entropy(y_right)

    p_left = len(y_left)/len(y)
    p_right = len(y_right)/len(y)

    H = p_left*H_left + p_right*H_right

    return H

def find_optimal_split(X, y):
    cand_split_list = []
    for feature_idx in range(X.shape[1]):
        x_values = np.sort(X[:, feature_idx])
        # midpoints of all x_values
        threshold_list = (x_values[:-1] + x_values[1:])/2.
        for threshold in threshold_list:
            entropy = calc_split_entropy(feature_idx, threshold, X, y)
            cand_split = Split(feature_idx = feature_idx,
                               threshold = threshold,
                               entropy = entropy)
            cand_split_list.append(cand_split)
    optimal_split = min(cand_split_list, key=lambda split: split.entropy)
    return optimal_split

def numberGenerator():
    number = 0
    while True:
        yield number
        number += 1
