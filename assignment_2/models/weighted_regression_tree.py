"""
This module implements a custom regression tree for tabular data, supporting sample
weights for robust regression.
"""


import numpy as np


class WeightedRegressionTree:
    """
    Custom regression tree implementation with support for weighted data
    intended to mimic the signature of the Scikit-learn tree regressor.
    Supports hyperparameters for maximum tree depth, minimum samples per
    split and per leaf.
    """
    def __init__(
            self,
            max_depth: int=6,
            min_samples_split: int=2,
            min_samples_leaf: int=1,
            random_state: int=42):
        """
        Initialize the CustomRegressionTree.

        Args:
            max_depth (int): Maximum depth of tree.
            min_samples_split (int): Minimum samples required to split internal nodes.
            min_samples_leaf (int): Minimum samples required on leaf nodes.
            random_state (int): Used only to match Scikit-learn parameters.

        Raises:
            ValueError: If any of the parameters are not valid positive integers.
        """
        # Enforce max_depth is positive integer
        if not isinstance(max_depth, int) or max_depth <= 0:
            raise ValueError("max_depth must be integer greater than 0")

        # Enforce min_samples_split is positive ingeger
        if not isinstance(min_samples_split, int) or min_samples_split <= 0:
            raise ValueError("min_samples_split must be integer greater than 0")

        # Enforce min_samples_leaf is positive ingeger
        if not isinstance(min_samples_leaf, int) or min_samples_leaf <= 0:
            raise ValueError("min_samples_leaf must be integer greater than 0")

        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.tree = None

        # To match Scikit-learn regression tree
        _ = random_state


    def fit(
            self,
            X: np.ndarray,
            y: np.ndarray,
            w: np.ndarray=None) -> None:
        """
        Builds a regression tree from the training set (X, y).
        Optionally, the fit can be weighted.

        Args:
            X (np.ndarray): Training input samples.
            y (np.ndarray): Target values.
            w (np.ndarray, optional): Weights for individual samples.
        """
        if w is None:
            w = np.ones(X.shape[0])
        self.n_features_ = X.shape[1]
        self.tree = self.recursive_build_tree(X, y, w, recurse_depth=0)


    def predict(
            self,
            X: np.ndarray) -> np.array:
        """
        Predict target values for given input samples using the fitted tree.

        Args:
            X (np.ndarray): Feature matrix of shape (n_samples, n_features).

        Returns:
            np.ndarray: Predicted target values of shape (n_samples, ).
        """
        return np.array([self.predict_row(row, self.tree) for row in X])


    def recursive_build_tree(
            self,
            X: np.array,
            y: np.array,
            w: np.ndarray,
            recurse_depth: int) -> dict:
        """
        Recursively build a regression tree node and its children.

        Args:
            X (np.ndarray): Input samples for the current node.
            y (np.ndarray): Target values for the current node.
            w (np.ndarray, optional): Weights for individual samples.
            recurse_depth (int): Current recursion depth (tree depth).

        Returns:
            dict: Nested dictionaries representing the current node and its subtree.
        """
        node = {}
        n_samples, n_features = X.shape
        
        node["value"] = np.average(y, weights=w)
        
        # Check current depth is less than maximum depth
        if recurse_depth >= self.max_depth:
            node["leaf"] = True
            return node
        
        # Check minimum samples required for split are available
        if n_samples < self.min_samples_split:
            node["leaf"] = True
            return node

        # Checks if the amount of labels is 1
        if (np.unique(y).size == 1):
            node["leaf"] = True
            return node

        # Find best split point for best feature
        best_feature, best_threshold, best_loss, left_idx, right_idx = self.best_split(X, y, w)

        # Stop splitting if there is no best feature
        if best_feature is None:
            node["leaf"] = True
            return node

        # Check tentative leaf node number of samples meets required minimum
        if left_idx.sum() < self.min_samples_leaf or right_idx.sum() < self.min_samples_leaf:
            node["leaf"] = True
            return node

        node["leaf"] = False
        node["feature"] = best_feature
        node["threshold"] = best_threshold

        # Build the next layer of the tree with an additional depth
        node["left"] = self.recursive_build_tree(
            X[left_idx],
            y[left_idx],
            w[left_idx],
            recurse_depth + 1)
        node["right"] = self.recursive_build_tree(
            X[right_idx],
            y[right_idx],
            w[right_idx],
            recurse_depth + 1)

        return node


    def best_split(
            self,
            X: np.ndarray,
            y: np.ndarray,
            w: np.ndarray) -> tuple:
        """
        Identify the best feature and threshold to split the given data to
        minimize mean squared error (variance).

        Args:
            X (np.ndarray): Feature matrix for the current node.
            y (np.ndarray): Target values for the current node.
            w (np.ndarray, optional): Weights for individual samples.

        Returns:
            tuple:
                best_feature (int or None): Index of the best splitting feature.
                best_threshold (int or None): Threshold for best split.
                best_loss (float or None): Loss value for the best split.
                left_idx (np.ndarray or None): Boolean index for left child samples.
                right_idx (np.ndarray or None): Boolean indes for right child samples.
        """
        n_samples, n_features = X.shape

        # Stop if there is only one sample
        if n_samples <= 1:
            return None, None, None, None, None

        best_loss = np.inf
        best_feature = None
        best_threshold = None

        # Checks all features
        for feature in range(n_features):
            X_col = X[:, feature]
            
            sorted_idx = np.argsort(X_col)
            X_sorted = X_col[sorted_idx]
            y_sorted = y[sorted_idx]
            w_sorted = w[sorted_idx]

            cum_w = np.cumsum(w_sorted)
            cum_w_inv = np.sum(w_sorted) - cum_w

            cum_yw = np.cumsum(y_sorted * w_sorted)
            cum_y2w = np.cumsum(y_sorted**2 * w_sorted)

            # Check all partitions
            for i in range(1, n_samples):
                if X_sorted[i] == X_sorted[i - 1]:
                    continue

                wL = cum_w[i-1]
                yL_sum = cum_yw[i-1]
                yL2_sum = cum_y2w[i-1]

                wR = cum_w[-1] - wL
                yR_sum = cum_yw[-1] - yL_sum
                yR2_sum = cum_y2w[-1] - yL2_sum

                if wL < 1e-9 or wR < 1e-9:
                    continue

                left_loss = yL2_sum - yL_sum * yL_sum / wL
                right_loss = yR2_sum - yR_sum * yR_sum / wR
                loss = left_loss + right_loss

                # Find best loss
                if loss < best_loss:
                    best_loss = loss
                    best_feature = feature
                    best_threshold = (X_sorted[i] + X_sorted[i - 1]) / 2.0

        if best_feature is None:
            return None, None, None, None, None

        left_idx = X[:, best_feature] <= best_threshold
        right_idx = ~left_idx

        return best_feature, best_threshold, best_loss, left_idx, right_idx


    def predict_row(
            self,
            row: np.array,
            node: dict) -> float:
        """
        Predict the target values for a single input sample using the trained tree.

        Args:
            row (np.ndarray): Single input sample (1D array of features).
            node (dict): The current node/subtree in the regression tree.

        Returns:
            float: The predicted target value for the sample.
        """
        if node.get("leaf", False):
            return node["value"]

        if row[node["feature"]] <= node["threshold"]:
            return self.predict_row(row, node["left"])
        else:
            return self.predict_row(row, node["right"])