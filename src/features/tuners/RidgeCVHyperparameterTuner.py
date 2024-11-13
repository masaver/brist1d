from sklearn.linear_model import RidgeCV
from skopt.space import Real, Integer, Categorical

from src.features.tuners.BaseHyperparameterTuner import BaseHyperparameterTuner

param_spaces = {
    'default': {
        'alphas': Real(1e-4, 1.0, prior='log-uniform'),  # Regularization strength; range of values to search over
        'max_iter': Integer(500, 5000),  # Number of iterations for convergence
        'tol': Real(1e-5, 1e-2, prior='log-uniform'),  # Tolerance for optimization
        'cv': Integer(3, 10),  # Number of cross-validation folds
        'normalize': Categorical([True, False]),  # Whether to normalize the input data
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
