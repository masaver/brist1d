from xgboost import XGBRegressor
from skopt.space import Integer, Real

from src.features.tuners.BaseHyperparameterTuner import BaseHyperparameterTuner

param_spaces = {
    'default': {
        'n_estimators': Integer(100, 300),  # Common range for boosting rounds
        'learning_rate': Real(0.03, 0.1, 'log-uniform'),  # Standard learning rates
        'max_depth': Integer(3, 7),  # Default depth range that balances depth and overfitting
        'min_child_weight': Integer(1, 5),  # Moderate range to allow small variations in regularization
        'subsample': Real(0.7, 0.85),  # Light subsampling for diversity while preserving data
        'colsample_bytree': Real(0.7, 0.85),  # Common column sampling range
        'gamma': Integer(0, 5),  # Small gamma range for light regularization
        'alpha': Real(1e-3, 0.1, 'log-uniform'),  # Default L1 regularization
        'lambda': Real(1e-3, 0.1, 'log-uniform'),  # Default L2 regularization
    },
    'deep': {
        'n_estimators': Integer(50, 1000),  # Number of boosting rounds
        'learning_rate': Real(0.001, 0.3, 'log-uniform'),  # Step size shrinkage, log-uniform for finer control
        'max_depth': Integer(3, 15),  # Maximum tree depth for base learners
        'min_child_weight': Integer(1, 10),  # Minimum sum of instance weight (hessian) needed in a child
        'subsample': Real(0.5, 1.0),  # Subsample ratio of the training instances
        'colsample_bytree': Real(0.3, 1.0),  # Subsample ratio of columns when constructing each tree
        'gamma': Integer(0, 10),  # Minimum loss reduction required to make a further partition on a leaf node
        'alpha': Real(1e-10, 10.0, 'log-uniform'),  # L1 regularization term on weights
        'lambda': Real(1e-10, 10.0, 'log-uniform'),  # L2 regularization term on weights
        'scale_pos_weight': Integer(1, 100),  # Controls balance of positive and negative weights, often used for imbalanced classes
    },
    'fast': {
        'n_estimators': [100, 200],  # Limited boosting rounds for quick training
        'learning_rate': [0.05, 0.1],  # Common default learning rates
        'max_depth': [3, 5],  # Shallow trees to keep training fast and avoid overfitting
        'min_child_weight': [1, 3],  # Minimal regularization range
        'subsample': [0.8],  # Fixed sampling value; often effective
        'colsample_bytree': [0.8],  # Fixed column sampling
        'gamma': [0, 1],  # Light regularization for basic pruning
    }
}


class XGBHyperparameterTuner(BaseHyperparameterTuner):
    __name__ = 'XGBRegressor'

    @staticmethod
    def regressor() -> XGBRegressor:
        return XGBRegressor(objective='reg:squarederror', random_state=42, tree_method='hist')

    @staticmethod
    def param_space(search_space: str | None) -> dict | None:
        if search_space is None:
            return None
        return param_spaces[search_space] if search_space in param_spaces.keys() else param_spaces['default']

    def fit(self, X, y):
        super().fit(X=X, y=y)
