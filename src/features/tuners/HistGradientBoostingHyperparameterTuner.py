from sklearn.ensemble import HistGradientBoostingRegressor
from skopt.space import Integer, Categorical, Real

from src.features.tuners.BaseHyperparameterTuner import BaseHyperparameterTuner

param_spaces = {
    'default': {
        'learning_rate': Real(0.01, 0.2, prior='log-uniform'),  # Learning rate (small values for smooth learning)
        'max_iter': Integer(100, 500),  # Number of boosting iterations
        'max_depth': Integer(3, 15),  # Maximum depth of each tree
        'min_samples_leaf': Integer(10, 50),  # Minimum samples in each leaf to prevent overfitting
        'max_leaf_nodes': Integer(15, 31),  # Maximum number of leaves in each tree
        'l2_regularization': Real(1e-5, 1.0, prior='log-uniform'),  # L2 regularization strength
        'max_bins': Integer(128, 255),  # Maximum number of bins to use for continuous values
        'early_stopping': Categorical([True, False]),  # Use early stopping to prevent overfitting
        'scoring': Categorical(['neg_mean_squared_error']),  # Choice of scoring metric
    }
}


class HistGradientBoostingHyperparameterTuner(BaseHyperparameterTuner):
    __name__ = 'HistGradientBoostingRegressor'

    @staticmethod
    def regressor() -> HistGradientBoostingRegressor:
        return HistGradientBoostingRegressor()

    @staticmethod
    def param_space(search_space: str | None) -> dict | None:
        if search_space is None:
            return None
        return param_spaces[search_space] if search_space in param_spaces.keys() else param_spaces['default']

    def fit(self, X, y):
        super().fit(X=X, y=y)
