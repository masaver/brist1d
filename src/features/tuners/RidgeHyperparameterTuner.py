from sklearn.linear_model import Ridge
from skopt.space import Real

from src.features.tuners.BaseHyperparameterTuner import BaseHyperparameterTuner

param_spaces = {
    'default': {
        'alpha': Real(1e-4, 10.0, prior='log-uniform'),  # Regularization strength; range of values to search over
    }
}


class RidgeHyperparameterTuner(BaseHyperparameterTuner):
    __name__ = 'RidgeRegressor'

    @staticmethod
    def regressor() -> Ridge:
        return Ridge()

    @staticmethod
    def param_space(search_space: str | None) -> dict | None:
        if search_space is None:
            return None
        return param_spaces[search_space] if search_space in param_spaces.keys() else param_spaces['default']

    def fit(self, X, y):
        super().fit(X=X, y=y)
