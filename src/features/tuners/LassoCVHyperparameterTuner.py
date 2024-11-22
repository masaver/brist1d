from sklearn.linear_model import LassoCV
from skopt.space import Integer, Real

from src.features.tuners.BaseHyperparameterTuner import BaseHyperparameterTuner

param_spaces = {
    'default': {
        'eps': Real(1e-5, 1e-2, prior='log-uniform'),  # Length of the path
        'n_alphas': Integer(100, 1000),  # Number of alphas along the regularization path
        'max_iter': Integer(1000, 5000),  # The maximum number of iterations
        'tol': Real(1e-5, 1e-2, prior='log-uniform'),  # The tolerance for the optimization
        'cv': Integer(2, 10),  # Number of folds in the cross
        'n_jobs': [-1],  # Number of jobs to run in parallel
    },
}


class LassoCVHyperparameterTuner(BaseHyperparameterTuner):
    __name__ = LassoCV.__name__

    @staticmethod
    def regressor():
        return LassoCV()

    @staticmethod
    def param_space(search_space: str | None) -> dict | None:
        if search_space is None:
            return None
        return param_spaces[search_space] if search_space in param_spaces.keys() else param_spaces['default']

    def fit(self, X_train, y_train, X_test=None, y_test=None):
        super().fit(X_train, y_train, X_test, y_test)
