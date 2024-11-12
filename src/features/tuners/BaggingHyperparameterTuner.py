from sklearn.ensemble import BaggingRegressor
from sklearn.tree import DecisionTreeRegressor

from src.features.tuners.BaseHyperparameterTuner import BaseHyperparameterTuner

param_spaces = {
    'default': {
        'n_estimators': (20, 100),  # Common range for boosting rounds
        'max_samples': (0.5, 1.0),  # Fraction of samples to use for training
        'max_features': (0.5, 1.0),  # Fraction of features to use for training
    },
    'wide': {
        'n_estimators': (10, 200),  # Wide range for boosting rounds
        'max_samples': (0.1, 1.0),  # Wide range for sample fraction
        'max_features': (0.1, 1.0),  # Wide range for feature fraction
    }
}


class BaggingHyperparameterTuner(BaseHyperparameterTuner):
    __name__ = 'BaggingRegressor'

    @staticmethod
    def regressor():
        return BaggingRegressor(estimator=DecisionTreeRegressor())

    @staticmethod
    def param_space(search_space: str) -> dict:
        return param_spaces[search_space] if search_space in param_spaces.keys() else param_spaces['default']

    def fit(self, X, y):
        super().fit(X=X, y=y)
