from sklearn.linear_model import Lasso
from skopt.space import Integer, Real

from src.features.tuners.BaseHyperparameterTuner import BaseHyperparameterTuner

param_spaces = {
    'default': {
        'alpha': Real(1e-4, 10.0, prior='log-uniform'),  # Regularization strength; range of values to search over
    }
}


class LassoHyperparameterTuner(BaseHyperparameterTuner):
    __name__ = Lasso.__name__

    @staticmethod
    def regressor():
        return Lasso()

    @staticmethod
    def param_space(search_space: str | None) -> dict | None:
        if search_space is None:
            return None
        return param_spaces[search_space] if search_space in param_spaces.keys() else param_spaces['default']

    def fit(self, X_train, y_train, X_test=None, y_test=None):
        super().fit(X_train, y_train, X_test, y_test)
