from typing import Literal

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from skopt import BayesSearchCV
from skopt.space import Integer, Real, Categorical
from xgboost import XGBRegressor
from sklearn.metrics import root_mean_squared_error, r2_score, PredictionErrorDisplay

param_spaces = {
    'custom': {
        'n_estimators': Integer(100, 1000),  # Number of boosting rounds
        'learning_rate': Real(0.001, 0.3, prior='log-uniform'),  # Learning rate with log-uniform distribution
        'max_depth': Integer(3, 15),  # Tree depth
        'min_child_weight': Integer(1, 5),  # Minimum child weight for regularization
        'subsample': Real(0.7, 0.85),  # Subsampling rate
        'colsample_bytree': Real(0.7, 0.85),  # Column sampling rate
        'gamma': Real(0, 5),  # Minimum loss reduction for partitioning
        'alpha': Real(1e-10, 10.0, prior='log-uniform'),  # L1 regularization term with log-uniform distribution
        'lambda': Real(1e-10, 10.0, prior='log-uniform'),  # L2 regularization term with log-uniform distribution
        'scale_pos_weight': Integer(1, 100),  # Balance for positive and negative class weights
    },
    'default': {
        'n_estimators': Integer(100, 300),  # Number of boosting rounds
        'learning_rate': Real(0.03, 0.1, prior='log-uniform'),  # Learning rate with log-uniform distribution
        'max_depth': Integer(3, 7),  # Tree depth
        'min_child_weight': Integer(1, 5),  # Minimum child weight for regularization
        'subsample': Real(0.7, 0.85),  # Subsampling rate
        'colsample_bytree': Real(0.7, 0.85),  # Column sampling rate
        'gamma': Real(0, 3),  # Minimum loss reduction for partitioning
        'alpha': Real(1e-3, 0.1, prior='log-uniform'),  # L1 regularization term with log-uniform distribution
        'lambda': Real(1e-3, 0.1, prior='log-uniform'),  # L2 regularization term with log-uniform distribution
    },
    'deep': {
        'n_estimators': Integer(50, 1000),  # Number of boosting rounds
        'learning_rate': Real(0.001, 0.3, prior='log-uniform'),  # Learning rate with log-uniform distribution
        'max_depth': Integer(3, 15),  # Maximum tree depth
        'min_child_weight': Integer(1, 10),  # Minimum child weight (regularization)
        'subsample': Real(0.5, 1.0),  # Subsampling ratio
        'colsample_bytree': Real(0.3, 1.0),  # Column sampling ratio
        'gamma': Real(0, 10),  # Minimum loss reduction required for partitioning
        'alpha': Real(1e-10, 10.0, prior='log-uniform'),  # L1 regularization term with log-uniform distribution
        'lambda': Real(1e-10, 10.0, prior='log-uniform'),  # L2 regularization term with log-uniform distribution
        'scale_pos_weight': Integer(1, 100),  # Class weight balancing for imbalanced datasets
    },
    'fast': {
        'n_estimators': Categorical([100, 200]),  # Limited number of boosting rounds for quick training
        'learning_rate': Categorical([0.05, 0.1]),  # Fixed learning rates for fast tuning
        'max_depth': Categorical([3, 5]),  # Shallow trees to keep training fast
        'min_child_weight': Categorical([1, 3]),  # Minimal regularization range
        'subsample': Categorical([0.8]),  # Fixed subsample ratio
        'colsample_bytree': Categorical([0.8]),  # Fixed column subsampling ratio
        'gamma': Categorical([0, 1]),  # Light regularization for pruning
    }
}


class XGBHyperparameterTuner:
    _search_space: str
    _transform_target: Literal['log', 'log1p', 'sqrt', 'None'] | None
    _best_model: XGBRegressor | None
    _best_params: dict | None
    _y_train: pd.Series | None
    _y_pred: pd.Series | None
    __name__ = 'XGBRegressor'

    def __init__(self, search_space='default', transform_target: Literal['log', 'log1p', 'sqrt', 'None'] = None):
        self._search_space = param_spaces[search_space] if search_space in param_spaces.keys() else param_spaces['default']
        self._transform_target = transform_target

    def fit(self, X, y):
        self._y_train = y

        np.int = int
        search_cv = BayesSearchCV(
            estimator=XGBRegressor(objective='reg:squarederror', random_state=42, tree_method='hist'),
            search_spaces=self._search_space,
            n_iter=30,
            scoring='neg_mean_squared_error',
            cv=5,
            n_jobs=-1,
            random_state=42
        )

        if self._transform_target is not None:
            if self._transform_target == 'log':
                y = np.log(y)
            elif self._transform_target == 'log1p':
                y = np.log1p(y)
            elif self._transform_target == 'sqrt':
                y = np.sqrt(y)

        search_cv.fit(X=X, y=y)

        regressor = XGBRegressor(**search_cv.best_params_, objective='reg:squarederror', random_state=42, tree_method='hist')
        regressor.fit(X=X, y=y)

        self._best_model = regressor
        self._best_params = search_cv.best_params_

        y_pred = regressor.predict(X=X)
        if self._transform_target is not None:
            if self._transform_target == 'log':
                y_pred = np.exp(y_pred)
            elif self._transform_target == 'log1p':
                y_pred = np.expm1(y_pred)
            elif self._transform_target == 'sqrt':
                y_pred = y_pred ** 2
        self._y_pred = y_pred

    def get_params(self):
        if self._best_params is None:
            raise ValueError('No model has been fitted yet')

        return self._best_params

    def get_r2_score(self):
        if self._y_pred is None or self._y_train is None:
            raise ValueError('No model has been fitted yet')
        return r2_score(y_true=self._y_train, y_pred=self._y_pred)

    def get_rmse(self):
        if self._y_pred is None or self._y_train is None:
            raise ValueError('No model has been fitted yet')
        return root_mean_squared_error(y_true=self._y_train, y_pred=self._y_pred)

    def get_best_model(self):
        if self._best_model is None:
            raise ValueError('No model has been fitted yet')
        return self._best_model

    def show_chart(self):
        if self._best_model is None:
            raise ValueError('No model has been fitted yet')

        fig, axs = plt.subplots(ncols=2, figsize=(8, 4))
        PredictionErrorDisplay.from_predictions(
            y_true=self._y_train,
            y_pred=self._y_pred,
            kind="actual_vs_predicted",
            subsample=100,
            ax=axs[0],
            random_state=0,
        )
        axs[0].set_title("Actual vs. Predicted values")
        PredictionErrorDisplay.from_predictions(
            y_true=self._y_train,
            y_pred=self._y_pred,
            kind="residual_vs_predicted",
            subsample=100,
            ax=axs[1],
            random_state=0,
        )
        axs[1].set_title("Residuals vs. Predicted Values")
        fig.suptitle("Plotting cross-validated predictions")
        plt.tight_layout()
        plt.show()
