from sklearn.linear_model import Lasso, ElasticNet
from skopt.space import Integer, Real

from src.features.tuners.BaseHyperparameterTuner import BaseHyperparameterTuner

param_spaces = {
    'default': {
        'alpha': Real(1e-4, 10.0, prior='log-uniform'),
        'l1_ratio': Real(0.0, 1.0, prior='uniform'),
    },
    'custom': {
        'alpha': Real(1e-6, 1e+1, prior='log-uniform'),  # Regularization strength (log-uniform distribution)
        'l1_ratio': Real(0.0, 1.0),  # Mix between Lasso (L1) and Ridge (L2)
        'max_iter': Integer(100, 10000),  # Maximum number of iterations
        'tol': Real(1e-5, 1e-1),  # Tolerance for optimization (smaller means more precise)
    }
}


class ElasticNetHyperparameterTuner(BaseHyperparameterTuner):
    __name__ = ElasticNet.__name__

    @staticmethod
    def regressor():
        return ElasticNet()

    @staticmethod
    def param_space(search_space: str | None) -> dict | None:
        if search_space is None:
            return None
        return param_spaces[search_space] if search_space in param_spaces.keys() else param_spaces['default']

    def fit(self, X_train, y_train, X_test=None, y_test=None):
        super().fit(X_train, y_train, X_test, y_test)
