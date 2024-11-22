from sklearn.linear_model import Lasso, ElasticNet
from skopt.space import Integer, Real

from src.features.tuners.BaseHyperparameterTuner import BaseHyperparameterTuner

param_spaces = {
    'default': {
        'alpha': Real(1e-4, 10.0, prior='log-uniform'),
        'l1_ratio': Real(0.0, 1.0, prior='uniform'),
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
