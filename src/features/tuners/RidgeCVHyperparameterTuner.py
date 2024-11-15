from sklearn.linear_model import RidgeCV
from skopt.space import Real, Integer, Categorical

from src.features.tuners.BaseHyperparameterTuner import BaseHyperparameterTuner

param_spaces = {
    'default': {
        'alphas': Categorical([0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]),  # Array of alpha values to try
    }
}


class RidgeCVHyperparameterTuner(BaseHyperparameterTuner):
    __name__ = 'RidgeCVRegressor'

    @staticmethod
    def regressor() -> RidgeCV:
        return RidgeCV()

    @staticmethod
    def param_space(search_space: str | None) -> dict | None:
        if search_space is None:
            return None
        return param_spaces[search_space] if search_space in param_spaces.keys() else param_spaces['default']

    def fit(self, X, y):
        super().fit(X=X, y=y)
