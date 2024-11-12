from sklearn.ensemble import ExtraTreesRegressor
from skopt.space import Integer, Real, Categorical

from src.features.tuners.BaseHyperparameterTuner import BaseHyperparameterTuner

param_spaces = {

    'default': {
        'max_depth': Integer(1, 20),  # Depth of the tree, controlling complexity and overfitting
        'min_samples_split': Integer(2, 20),  # Minimum samples required to split an internal node
        'min_samples_leaf': Integer(1, 10),  # Minimum samples required in a leaf node
        'max_features': Real(0.1, 1.0),  # Fraction of features to consider at each split
    },

    'deep': {
        'n_estimators': Integer(100, 1000),                 # Wider range for number of trees
        'max_depth': Integer(5, 50),                        # Deeper trees if desired
        'min_samples_split': Integer(2, 50),                # Higher minimum split values
        'min_samples_leaf': Integer(1, 50),                 # Higher range for min samples at leaf
        'max_features': Categorical(['sqrt', 'log2', None]),  # Standard feature options
        'bootstrap': Categorical([True, False]),            # Bootstrap sampling option
        'max_leaf_nodes': Integer(10, 1000),                # Maximum number of leaf nodes per tree
        'min_impurity_decrease': Real(0.0, 0.5, 'uniform'), # Minimum impurity decrease for splitting
        'ccp_alpha': Real(0.0, 0.1, 'uniform')              # Complexity parameter for Minimal Cost-Complexity Pruning
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
