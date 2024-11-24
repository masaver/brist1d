from sklearn.linear_model import LassoLarsIC
from skopt.space import Integer, Real, Categorical

from src.features.tuners.BaseHyperparameterTuner import BaseHyperparameterTuner

param_spaces = {
    'default': {
        'criterion': Categorical(['aic', 'bic']),  # Criterion for model selection
        'eps': Real(1e-5, 1e-1),  # Length of the path
        'max_iter': Integer(1000, 5000),  # Maximum number of iterations
        'noise_variance': Real(1e-6, 1e-2),  # Noise variance
        'positive': Categorical([True, False]),  # Restrict coefficients to be positive
    }
}


class LassoLarsICHyperparameterTuner(BaseHyperparameterTuner):
    __name__ = LassoLarsIC.__name__

    @staticmethod
    def regressor():
        return LassoLarsIC()

    @staticmethod
    def param_space(search_space: str | None) -> dict | None:
        if search_space is None:
            return None
        return param_spaces[search_space] if search_space in param_spaces.keys() else param_spaces['default']

    def fit(self, X_train, y_train, X_test=None, y_test=None):
        super().fit(X_train, y_train, X_test, y_test)
