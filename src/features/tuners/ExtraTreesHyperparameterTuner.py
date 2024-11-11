from sklearn.tree import ExtraTreeRegressor
from src.features.tuners.BaseHyperparameterTuner import BaseHyperparameterTuner

param_spaces = {
    'default': {
        'max_depth': (1, 20),  # Depth of the tree, controlling complexity and overfitting
        'min_samples_split': (2, 20),  # Minimum samples required to split an internal node
        'min_samples_leaf': (1, 10),  # Minimum samples required in a leaf node
        'max_features': (0.1, 1.0),  # Fraction of features to consider at each split
        'splitter': ['random', 'best'],  # Choose between best or random split at each node
    }
}


class ExtraTreesHyperparameterTuner(BaseHyperparameterTuner):
    __name__ = 'ExtraTreeRegressor'

    @staticmethod
    def regressor():
        return ExtraTreeRegressor()

    @staticmethod
    def param_space(search_space: str) -> dict:
        return param_spaces[search_space] if search_space in param_spaces.keys() else param_spaces['default']

    def fit(self, X, y):
        super().fit(X=X, y=y)
