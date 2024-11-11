import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import LarsCV
from skopt import BayesSearchCV
from sklearn.metrics import root_mean_squared_error, r2_score, PredictionErrorDisplay
from skopt.space import Integer, Real

param_spaces = {
    'default': {
        'eps': Real(1e-5, 1e-2, prior='log-uniform'),  # Length of the path
        'max_iter': Integer(1000, 5000),  # Max iterations for convergence
        'max_n_alphas': Integer(100, 1000),  # Maximum number of alphas along the regularization path
    }
}


class LassoCVHyperparameterTuner:
    _search_space: str
    _best_model: LarsCV | None
    _best_params: dict | None
    _y_train: pd.Series | None
    _y_pred: pd.Series | None
    __name__ = LarsCV.__name__

    def __init__(self, search_space='default'):
        self._search_space = param_spaces[search_space] if search_space in param_spaces.keys() else param_spaces['default']

    def fit(self, X, y):
        np.int = int
        search_cv = BayesSearchCV(
            estimator=LarsCV(),
            search_spaces=self._search_space,
            n_iter=30,
            scoring='neg_mean_squared_error',
            cv=5,
            n_jobs=-1,
        )
        search_cv.fit(X=X, y=y)

        regressor = LarsCV(**search_cv.best_params_)
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
