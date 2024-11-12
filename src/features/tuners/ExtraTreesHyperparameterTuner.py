from sklearn.ensemble import ExtraTreesRegressor
from skopt.space import Integer, Real

from src.features.tuners.BaseHyperparameterTuner import BaseHyperparameterTuner

param_spaces = {
    'default': {
        'n_estimators': Integer(10, 100),  # Number of trees in the forest
        'max_depth': Integer(3, 15),  # Maximum depth of each tree
        'min_samples_split': Integer(2, 10),  # Minimum samples to split a node
        'min_samples_leaf': Integer(1, 10),  # Minimum samples in each leaf
        'min_weight_fraction_leaf': Real(0, 0.5),  # Minimum weight fraction in a leaf
        'max_features': Real(0.8, 1.0),  # Fraction of features to use for training
        'max_leaf_nodes': Integer(15, 31),  # Maximum number of leaves in each tree
        'min_impurity_decrease': Real(0, 0.1),  # Minimum impurity decrease to split a node
        'random_state': [42],  # Random seed for reproducibility
    }
}


class ExtraTreesHyperparameterTuner(BaseHyperparameterTuner):
    __name__ = ExtraTreesRegressor.__name__

    @staticmethod
    def regressor():
        return ExtraTreesRegressor()

    @staticmethod
    def param_space(search_space: str) -> dict:
        return param_spaces[search_space] if search_space in param_spaces.keys() else param_spaces['default']

    def fit(self, X, y):
        super().fit(X=X, y=y)