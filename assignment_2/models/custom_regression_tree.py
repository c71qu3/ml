# -*- coding: utf-8 -*-
"""
Created on Sun Nov 30 19:46:59 2025

@author: mitre
"""

import numpy as np

class CustomRegressionTree:
    def __init__(self, max_depth: int = 6, min_samples_split: int = 2, min_samples_leaf: int = 1):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.tree = None

    def fit(self, X: np.array, y: np.array):
        self.n_features_ = X.shape[1]
        self.tree = self.recursive_build_tree(X, y, recurse_depth=0)

    def predict(self, X: np.array):
        return np.array([self.predict_row(row, self.tree) for row in X])

    def recursive_build_tree(self, X: np.array, y: np.array, recurse_depth: int):
        node = {}
        n_samples, n_features = X.shape
        
        node["value"] = y.mean()
        
        # Check if the maximum depth parameter is set, and if so, if the current depth is greater or equal than, stops splitting
        if (self.max_depth is not None and recurse_depth >= self.max_depth):
            node["leaf"] = True
            return node
        
        # Checks if the number of samples for the split is above the thresold, stops splitting
        if n_samples < self.min_samples_split:
            node["leaf"] = True
            return node

        # Checks if the amount of labels is 1, stops splitting
        if (np.unique(y).size == 1):
            node["leaf"] = True
            return node

        # For each feature, finds the best split point
        best_feature, best_threshold, best_loss, left_idx, right_idx = self.best_split(X, y)

        # If there is no best feature, stop splitting
        if best_feature is None:
            node["leaf"] = True
            return node

        # If the number of samples in either side after the split is less than the threshold, stops splitting
        if left_idx.sum() < self.min_samples_leaf or right_idx.sum() < self.min_samples_leaf:
            node["leaf"] = True
            return node

        node["leaf"] = False
        node["feature"] = best_feature
        node["threshold"] = best_threshold

        # Build the next layer of the tree with an additional depth
        node["left"] = self.recursive_build_tree(X[left_idx],  y[left_idx],  recurse_depth + 1)
        node["right"] = self.recursive_build_tree(X[right_idx], y[right_idx], recurse_depth + 1)

        return node

    def best_split(self, X: np.array, y: np.array):
        n_samples, n_features = X.shape
        # If there is only one sample, cannot find any split
        if n_samples <= 1:
            return None, None, None, None, None

        best_loss = np.inf
        best_feature = None
        best_threshold = None

        # Checks each feature, evaluates the loss on each possible split per feature and finds the one that minimizes the loss
        for feature in range(n_features):
            X_col = X[:, feature]
            
            sorted_idx = np.argsort(X_col)
            X_sorted = X_col[sorted_idx]
            y_sorted = y[sorted_idx]

            cum_sum = np.cumsum(y_sorted)
            cum_sq_sum = np.cumsum(y_sorted**2)

            for i in range(1, n_samples):
                if X_sorted[i] == X_sorted[i - 1]:
                    continue

                left_count = i
                right_count = n_samples - i

                left_sum = cum_sum[i - 1]
                left_sq_sum = cum_sq_sum[i - 1]

                right_sum = cum_sum[-1] - left_sum
                right_sq_sum = cum_sq_sum[-1] - left_sq_sum

                left_loss = left_sq_sum - (left_sum**2) / left_count
                right_loss = right_sq_sum - (right_sum**2) / right_count
                loss = left_loss + right_loss

                if loss < best_loss:
                    best_loss = loss
                    best_feature = feature
                    best_threshold = (X_sorted[i] + X_sorted[i - 1]) / 2.0

        if best_feature is None:
            return None, None, None, None, None

        left_idx = X[:, best_feature] <= best_threshold
        right_idx = ~left_idx

        return best_feature, best_threshold, best_loss, left_idx, right_idx

    def predict_row(self, row: np.array, node: dict):
        if node.get("leaf", False):
            return node["value"]

        if row[node["feature"]] <= node["threshold"]:
            return self.predict_row(row, node["left"])
        else:
            return self.predict_row(row, node["right"])