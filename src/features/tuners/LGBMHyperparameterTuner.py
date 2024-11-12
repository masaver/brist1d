from lightgbm import LGBMRegressor
from skopt.space import Integer, Real
from src.features.tuners.BaseHyperparameterTuner import BaseHyperparameterTuner

param_spaces = {
    'default': {
        'n_estimators': Integer(100, 1000),  # Number of boosting rounds
        'learning_rate': Real(0.01, 0.3, 'log-uniform'),  # Smaller values require more boosting rounds
        'max_depth': Integer(3, 15),  # Limits depth of the individual trees
        'num_leaves': Integer(20, 150),  # Number of leaves in each tree
        'min_child_samples': Integer(5, 50),  # Minimum data points needed in a leaf
        'min_child_weight': Real(0.001, 10.0, 'log-uniform'),  # Minimum sum of instance weight (hessian)
        'subsample': Real(0.5, 1.0, 'uniform'),  # Fraction of data to sample for each tree
        'colsample_bytree': Real(0.5, 1.0, 'uniform'),  # Fraction of features for each tree
        'reg_alpha': Real(1e-8, 10.0, 'log-uniform'),  # L1 regularization
        'reg_lambda': Real(1e-8, 10.0, 'log-uniform')  # L2 regularization
    },
    'deep': {
        'n_estimators': Integer(100, 2000),  # Number of boosting rounds (broader range)
        'learning_rate': Real(0.005, 0.3, 'log-uniform'),  # Finer control for learning rate
        'max_depth': Integer(2, 20),  # Expanded range for tree depth
        'num_leaves': Integer(10, 300),  # Broader range for number of leaves
        'min_child_samples': Integer(1, 100),  # Minimum data points in a leaf (broader range)
        'min_child_weight': Real(1e-4, 50.0, 'log-uniform'),  # Controls minimum sum of instance weight (hessian) in a leaf
        'subsample': Real(0.4, 1.0, 'uniform'),  # Broader range for row sampling
        'subsample_freq': Integer(0, 10),  # Frequency of subsampling
        'colsample_bytree': Real(0.4, 1.0, 'uniform'),  # Expanded range for column sampling for each tree
        'colsample_bynode': Real(0.4, 1.0, 'uniform'),  # Column sampling at each split
        'colsample_bylevel': Real(0.4, 1.0, 'uniform'),  # Column sampling at each level
        'reg_alpha': Real(1e-10, 100.0, 'log-uniform'),  # Expanded range for L1 regularization
        'reg_lambda': Real(1e-10, 100.0, 'log-uniform'),  # Expanded range for L2 regularization
        'max_bin': Integer(100, 500),  # Number of bins used for splitting features
        'min_split_gain': Real(0.0, 1.0, 'uniform')  # Minimum gain to make a split (regularization)
    }
}


class LGBMHyperparameterTuner(BaseHyperparameterTuner):
    __name__ = LGBMRegressor.__name__

    @staticmethod
    def regressor() -> LGBMRegressor:
        return LGBMRegressor(verbose=-1, random_state=42)

    @staticmethod
    def param_space(search_space: str) -> dict:
        return param_spaces[search_space] if search_space in param_spaces.keys() else param_spaces['default']

    def fit(self, X, y):
        super().fit(X=X, y=y)
