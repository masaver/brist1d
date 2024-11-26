import numpy as np
from sklearn.model_selection import BaseCrossValidator, ShuffleSplit


class CustomSplitter(BaseCrossValidator):
    def __init__(self, test_size=0.2, n_splits=1, random_state=None):
        """
        Custom splitter for BayesSearchCV.

        Parameters:
        - test_size (float): Proportion of the data with group 1 to use in the test set.
        - n_splits (int): Number of splits to generate.
        - random_state (int or None): Random seed for reproducibility.
        """
        self.test_size = test_size
        self.n_splits = n_splits
        self.random_state = random_state
        self._groups = None

    def fit(self, X, y=None, groups=None):
        """
        Store the group list provided during fitting.
        Parameters:
        - X: Input data (not used, included for API compatibility).
        - y: Target values (not used, included for API compatibility).
        - groups (array-like): List of group values (0, 1).
        """
        if groups is None:
            raise ValueError("Groups must be provided.")
        self._groups = np.asarray(groups)
        return self

    def split(self, X, y=None, groups=None):
        """
        Generate train/test splits.

        Parameters:
        - X: Input data.
        - y: Target values (not used).
        - groups (ignored, the stored groups from `fit` are used).

        Yields:
        - train_indices: Indices for the training set.
        - test_indices: Indices for the test set.
        """
        if self._groups is None:
            self.fit(X, groups=groups)

        # Indices for groups
        _group_0_indices = np.where(self._groups == 0)[0]
        _group_1_indices = np.where(self._groups == 1)[0]

        if len(_group_1_indices) == 0:
            raise ValueError("Group 1 has no samples to split. Please ensure that group 1 is not empty.")

        # Use ShuffleSplit for random splitting of group 1
        shuffle_split = ShuffleSplit(n_splits=self.n_splits, test_size=self.test_size, random_state=self.random_state)

        for train_1_indices, test_1_indices in shuffle_split.split(_group_1_indices):
            # Map the shuffled indices back to the original group_1_indices
            _train_1 = _group_1_indices[train_1_indices]
            _test_1 = _group_1_indices[test_1_indices]

            # Combine train indices
            train_indices = np.concatenate([_group_0_indices, _train_1])
            yield train_indices, _test_1

    def get_n_splits(self, X=None, y=None, groups=None):
        """
        Returns the number of splits.
        """
        return self.n_splits
