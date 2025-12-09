# -*- coding: utf-8 -*-
"""
Created on Mon Dec  1 04:05:34 2025

@author: mitre
"""

class CustomRegressionTree:
    def __init__(self, max_depth=6, min_samples_split=2, min_samples_leaf=1):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.tree = None

    def fit(self, X, y):
        # X is a list of lists, y is a list
        self.n_features_ = len(X[0])
        self.tree = self.recursive_build_tree(X, y, recurse_depth=0)

    def predict(self, X):
        return [self.predict_row(row, self.tree) for row in X]

    def recursive_build_tree(self, X, y, recurse_depth):
        node = {}
        n_samples = len(X)

        node["value"] = sum(y) / len(y)

        # If reached maximum depth, creates leafx
        if self.max_depth is not None and recurse_depth >= self.max_depth:
            node["leaf"] = True
            return node

        # If not enough samples for split, creates leaf
        if n_samples < self.min_samples_split:
            node["leaf"] = True
            return node

        # If all values of y are the same, creates leaf
        if len(set(y)) == 1:
            node["leaf"] = True
            return node

        # Find best split
        best_feature, best_threshold, left_idx, right_idx = \
            self.best_split(X, y)

        # If did not find improved split, creates leaf
        if best_feature is None:
            node["leaf"] = True
            return node

        # If split produces leaf with less than minimum size, creates leaf
        if len(left_idx) < self.min_samples_leaf or len(right_idx) < self.min_samples_leaf:
            node["leaf"] = True
            return node

        node["leaf"] = False
        node["feature"] = best_feature
        node["threshold"] = best_threshold

        # build subtrees with filtered data
        X_left  = [X[i] for i in left_idx]
        y_left  = [y[i] for i in left_idx]

        X_right = [X[i] for i in right_idx]
        y_right = [y[i] for i in right_idx]

        node["left"] = self.recursive_build_tree(X_left, y_left, recurse_depth + 1)
        node["right"] = self.recursive_build_tree(X_right, y_right, recurse_depth + 1)

        return node

    def best_split(self, X, y):
        n_samples = len(X)
        n_features = len(X[0])

        best_loss = float("inf")
        best_feature, best_threshold = None, None

        for feature in range(n_features):
            # Extract column
            X_col = [row[feature] for row in X]

            # Sort both X and Y by the feature
            sorted_indices = sorted(range(n_samples), key=lambda i: X_col[i])
            X_sorted = [X_col[i] for i in sorted_indices]
            y_sorted = [y[i] for i in sorted_indices]

            # Precompute cumulative sums
            cum_sum = []
            cum_square_sum = []
            running_sum = 0.0
            running_square_sum = 0.0

            # Use the optimized math identity for Sum of Squared Errors for efficiency
            # sum(y-y_mean)^2 = sum(y^2) - sum(y)^2/n
            # Reduces complexity from O(N^2) to O(N)
            for val in y_sorted:
                running_sum += val
                running_square_sum += val * val
                cum_sum.append(running_sum)
                cum_square_sum.append(running_square_sum)

            # Try all split positions, starting from after the first datapoint
            for i in range(1, n_samples):
                # If feature value is same as last feature value, don't need to recalculate
                if X_sorted[i] == X_sorted[i-1]:
                    continue

                left_count = i
                right_count = n_samples - i

                left_sum = cum_sum[i-1]
                left_square_sum = cum_square_sum[i-1]

                right_sum = cum_sum[-1] - left_sum
                right_square_sum = cum_square_sum[-1] - left_square_sum

                left_loss = left_square_sum - ((left_sum**2) / left_count)
                right_loss = right_square_sum - ((right_sum**2) / right_count)
                loss = left_loss + right_loss

                if loss < best_loss:
                    best_loss = loss
                    best_feature = feature
                    best_threshold = (X_sorted[i] + X_sorted[i-1]) / 2

        # no valid split found
        if best_feature is None:
            return None, None, None, None

        # final split masks (use original X)
        left_idx = []
        right_idx = []
        for i in range(n_samples):
            if X[i][best_feature] <= best_threshold:
                left_idx.append(i)
            else:
                right_idx.append(i)

        return best_feature, best_threshold, left_idx, right_idx

    def predict_row(self, row, node):
        if node.get("leaf", False):
            return node["value"]

        if row[node["feature"]] <= node["threshold"]:
            return self.predict_row(row, node["left"])
        else:
            return self.predict_row(row, node["right"])