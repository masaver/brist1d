import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.ensemble import HistGradientBoostingRegressor
from skopt import BayesSearchCV
from skopt.space import Integer, Categorical, Real
from sklearn.metrics import root_mean_squared_error, r2_score, PredictionErrorDisplay

param_spaces = {
    'default': {
        'learning_rate': Real(0.01, 0.2, prior='log-uniform'),  # Learning rate (small values for smooth learning)
        'max_iter': Integer(100, 500),  # Number of boosting iterations
        'max_depth': Integer(3, 15),  # Maximum depth of each tree
        'min_samples_leaf': Integer(10, 50),  # Minimum samples in each leaf to prevent overfitting
        'max_leaf_nodes': Integer(15, 31),  # Maximum number of leaves in each tree
        'l2_regularization': Real(1e-5, 1.0, prior='log-uniform'),  # L2 regularization strength
        'max_bins': Integer(128, 255),  # Maximum number of bins to use for continuous values
        'early_stopping': Categorical([True, False]),  # Use early stopping to prevent overfitting
        'scoring': Categorical(['neg_mean_squared_error']),  # Choice of scoring metric
    }
}


class HistGradientBoostingHyperparameterTuner:
    _search_space: str
    _best_model: HistGradientBoostingRegressor | None
    _best_params: dict | None
    _y_train: pd.Series | None
    _y_pred: pd.Series | None
    __name__ = 'HistGradientBoostingRegressor'

    def __init__(self, search_space='default'):
        self._search_space = param_spaces[search_space] if search_space in param_spaces.keys() else param_spaces['default']

    def fit(self, X, y):
        np.int = int
        regressor = HistGradientBoostingRegressor()
        search_cv = BayesSearchCV(
            estimator=regressor,
            search_spaces=self._search_space,
            n_iter=30,
            scoring='neg_mean_squared_error',
            cv=5,
            n_jobs=-1,
            random_state=42
        )
        search_cv.fit(X=X, y=y)

        regressor = HistGradientBoostingRegressor(**search_cv.best_params_)
        regressor.fit(X=X, y=y)

        self._best_model = regressor
        self._best_params = search_cv.best_params_
        self._y_train = y
        self._y_pred = regressor.predict(X=X)

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
