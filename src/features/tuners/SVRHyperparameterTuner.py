from sklearn.svm import SVR
from skopt.space import Integer, Categorical, Real

from src.features.tuners.BaseHyperparameterTuner import BaseHyperparameterTuner

param_spaces = {
    'default': {
        'C': Real(1e-3, 1e3, prior='log-uniform'),  # Regularization parameter
        'epsilon': Real(1e-4, 1.0, prior='log-uniform'),  # Width of the epsilon-tube
        'kernel': Categorical(['linear', 'poly', 'rbf', 'sigmoid']),  # Kernel type
        'degree': Integer(2, 5),  # Degree of the polynomial kernel (if 'poly')
        'gamma': Categorical(['scale', 'auto']),  # Kernel coefficient for 'rbf', 'poly', and 'sigmoid'
        'coef0': Real(0.0, 10.0),  # Independent term in kernel function (if 'poly' or 'sigmoid')
        'tol': Real(1e-5, 1e-2, prior='log-uniform'),  # Tolerance for stopping criterion
        'shrinking': Categorical([True, False]),  # Whether to use shrinking heuristic
        'max_iter': Integer(1000, 5000),  # Maximum number of iterations (-1 for no limit)
    },
    'custom': {
        'C': Real(1e-2, 100, prior='log-uniform'),  # Narrowed range to reduce overfitting risk
        'epsilon': Real(1e-4, 1.0, prior='log-uniform'),  # Keep original range
        'kernel': Categorical(['linear', 'poly', 'rbf']),  # Removed 'sigmoid'
        'degree': Integer(2, 3),  # Focus on simpler polynomial models
        'gamma': Categorical(['scale', 'auto']),  # No change
        'coef0': Real(0.0, 5.0),  # Narrower range for practicality
        'tol': Real(1e-5, 1e-2, prior='log-uniform'),  # No change
        'shrinking': Categorical([True, False]),  # No change
        'max_iter': Integer(-1, 5000),  # Added -1 for no iteration limit
    }
}


class SVRHyperparameterTuner(BaseHyperparameterTuner):
    __name__ = SVR.__name__

    @staticmethod
    def regressor() -> SVR:
        return SVR()

    @staticmethod
    def param_space(search_space: str | None) -> dict | None:
        if search_space is None:
            return None
        return param_spaces[search_space] if search_space in param_spaces.keys() else param_spaces['default']

    def fit(self, X_train, y_train, X_test=None, y_test=None):
        super().fit(X_train, y_train, X_test, y_test)
