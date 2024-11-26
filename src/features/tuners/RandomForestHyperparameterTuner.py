from sklearn.ensemble import RandomForestRegressor
from skopt.space import Integer, Categorical, Real

from src.features.tuners.BaseHyperparameterTuner import BaseHyperparameterTuner

param_spaces = {
    'default': {
        'n_estimators': Integer(100, 1000),  # Number of trees in the forest
        'max_depth': Integer(3, 30),  # Maximum depth of each tree
        'min_samples_split': Integer(2, 20),  # Minimum number of samples required to split an internal node
        'min_samples_leaf': Integer(1, 20),  # Minimum number of samples required to be at a leaf node
        'max_features': Categorical(['auto', 'sqrt', 'log2']),  # Number of features to consider for the best split
        'bootstrap': Categorical([True, False]),  # Whether to use bootstrap sampling
        'max_leaf_nodes': Integer(10, 200),  # Maximum number of leaf nodes
        'min_impurity_decrease': Real(0.0, 1e-3),  # Minimum impurity decrease for node splitting
        'ccp_alpha': Real(0.0, 1e-2),  # Complexity parameter for minimal cost-complexity pruning
    },
    'custom': {
        'n_estimators': Integer(100, 1000),  # Number of trees in the forest
        'max_depth': Integer(3, 30),  # Maximum depth of each tree
        'min_samples_split': Integer(2, 20),  # Minimum number of samples required to split an internal node
        'min_samples_leaf': Integer(1, 20),  # Minimum number of samples required to be at a leaf node
        'max_features': Categorical(['auto', 'sqrt', 'log2']),  # Number of features to consider for the best split
        'bootstrap': Categorical([True, False]),  # Whether to use bootstrap sampling
        'max_leaf_nodes': Integer(10, 200),  # Maximum number of leaf nodes
        'min_impurity_decrease': Real(0.0, 1e-3),  # Minimum impurity decrease for node splitting
        'ccp_alpha': Real(0.0, 1e-2),  # Complexity parameter for minimal cost-complexity pruning
    }
}


class RandomForestHyperparameterTuner(BaseHyperparameterTuner):
    __name__ = RandomForestRegressor.__name__

    @staticmethod
    def regressor() -> RandomForestRegressor:
        return RandomForestRegressor()

    @staticmethod
    def param_space(search_space: str | None) -> dict | None:
        if search_space is None:
            return None
        return param_spaces[search_space] if search_space in param_spaces.keys() else param_spaces['default']

    def fit(self, X_train, y_train, X_test=None, y_test=None):
        super().fit(X_train, y_train, X_test, y_test)
