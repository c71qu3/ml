import numpy as np
from .custom_regression_tree import CustomRegressionTree

class CustomRandomForest:
    """
    A custom Random Forest Regressor implementation from scratch.

    This algorithm builds an ensemble of CustomRegressionTree models. Each tree is
    trained on a bootstrapped sample of the data and considers a random subset of
    features for splitting, which helps in creating a robust and generalized model.
    """
    def __init__(self, n_trees: int = 100, max_depth: int = 10, 
                 min_samples_split: int = 2, min_samples_leaf: int = 1, 
                 max_features: str = 'sqrt', random_state: int = None):
        """
        Initializes the CustomRandomForest.

        Args:
            n_trees (int): The number of regression trees in the forest.
            max_depth (int): The maximum depth of each individual tree.
            min_samples_split (int): The minimum number of samples required to split an internal node.
            min_samples_leaf (int): The minimum number of samples required to be at a leaf node.
            max_features (str): The method to determine the number of features to consider for each tree.
                                Supported values: 'sqrt', 'log2', or an integer.
            random_state (int): Controls the randomness for bootstrapping and feature selection.
        """
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random_state = random_state
        self.trees = []
        self.feature_indices_list = []
        # Create a random number generator instance for reproducibility
        self._rng = np.random.default_rng(self.random_state)

    def fit(self, X: np.array, y: np.array):
        """
        Builds a forest of trees from the training set (X, y).

        Args:
            X (np.array): The training input samples.
            y (np.array): The target values.
        """
        n_samples, n_features = X.shape
        self._set_feature_sample_size(n_features)

        for _ in range(self.n_trees):
            # Create a bootstrapped sample (sampling with replacement)
            bootstrap_indices = self._rng.choice(n_samples, n_samples, replace=True)
            X_sample, y_sample = X[bootstrap_indices], y[bootstrap_indices]

            # Select a random subset of features
            feature_indices = self._rng.choice(n_features, self.n_features_sample, replace=False)
            self.feature_indices_list.append(feature_indices)

            # Initialize and train a regression tree on the sample and feature subset
            tree = CustomRegressionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf
            )
            tree.fit(X_sample[:, feature_indices], y_sample)
            self.trees.append(tree)

    def predict(self, X: np.array) -> np.array:
        """
        Predicts regression target for X.

        The prediction is the average of the predictions of all trees in the forest.

        Args:
            X (np.array): The input samples to predict.

        Returns:
            np.array: The predicted values.
        """
        # Matrix to store predictions from each tree
        predictions_matrix = np.zeros((len(self.trees), X.shape[0]))

        for i, tree in enumerate(self.trees):
            # Get the feature indices used for this specific tree
            feature_indices = self.feature_indices_list[i]
            X_subset = X[:, feature_indices]
            
            # Store the predictions of the tree
            predictions_matrix[i, :] = tree.predict(X_subset)

        # The final prediction is the average of all tree predictions
        return np.mean(predictions_matrix, axis=0)

    def _set_feature_sample_size(self, n_features: int):
        """
        Determines the number of features to use for each tree based on max_features.
        """
        if self.max_features == 'sqrt':
            self.n_features_sample = int(np.sqrt(n_features))
        elif self.max_features == 'log2':
            self.n_features_sample = int(np.log2(n_features))
        elif isinstance(self.max_features, int):
            self.n_features_sample = self.max_features
        else:
            self.n_features_sample = n_features
        
        # Ensure at least one feature is selected
        if self.n_features_sample < 1:
            self.n_features_sample = 1
