from sklearn.ensemble import ExtraTreesRegressor
from skopt.space import Integer, Real, Categorical

from src.features.tuners.BaseHyperparameterTuner import BaseHyperparameterTuner

param_spaces = {
    'default': {
        'n_estimators': Integer(10, 100),  # Number of trees in the forest
        'max_depth': Integer(3, 15),  # Maximum depth of each tree
        'min_samples_split': Integer(2, 10),  # Minimum samples to split a node
        'min_samples_leaf': Integer(1, 10),  # Minimum samples in each leaf
        'max_features': Categorical(['sqrt', 'log2', None]),  # Standard feature options
        'max_leaf_nodes': Integer(15, 31),  # Maximum number of leaves in each tree
        'min_impurity_decrease': Real(0, 0.1, 'uniform'),  # Minimum impurity decrease to split a node
        'random_state': [42],  # Random seed for reproducibility
    },
    'deep': {
        'n_estimators': Integer(100, 1000),  # Wider range for number of trees
        'max_depth': Integer(5, 50),  # Deeper trees if desired
        'min_samples_split': Integer(2, 50),  # Higher minimum split values
        'min_samples_leaf': Integer(1, 50),  # Higher range for min samples at leaf
        'max_features': Categorical(['sqrt', 'log2', None]),  # Standard feature options
        'bootstrap': Categorical([True, False]),  # Bootstrap sampling option
        'max_leaf_nodes': Integer(10, 1000),  # Maximum number of leaf nodes per tree
        'min_impurity_decrease': Real(0.0, 0.5, 'uniform'),  # Minimum impurity decrease for splitting
        'ccp_alpha': Real(0.0, 0.1, 'uniform')  # Complexity parameter for Minimal Cost-Complexity Pruning
    },
    'no': {
        'n_estimators': Integer(10, 100),
    }
}


class ExtraTreesHyperparameterTuner(BaseHyperparameterTuner):
    __name__ = ExtraTreesRegressor.__name__

    @staticmethod
    def regressor():
        return ExtraTreesRegressor(random_state=42)

    @staticmethod
    def param_space(search_space: str | None) -> dict | None:
        if search_space is None:
            return None
        return param_spaces[search_space] if search_space in param_spaces.keys() else param_spaces['default']

    def fit(self, X, y):
        super().fit(X=X, y=y)
