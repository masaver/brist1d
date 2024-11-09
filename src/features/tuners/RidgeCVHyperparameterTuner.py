import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import RidgeCV
from skopt import BayesSearchCV
from sklearn.metrics import root_mean_squared_error, r2_score, PredictionErrorDisplay
from skopt.space import Real, Integer, Categorical

param_spaces = {
    'default': {
        'alphas': Real(1e-4, 1.0, prior='log-uniform'),  # Regularization strength; range of values to search over
        'max_iter': Integer(500, 5000),  # Number of iterations for convergence
        'tol': Real(1e-5, 1e-2, prior='log-uniform'),  # Tolerance for optimization
        'cv': Integer(3, 10),  # Number of cross-validation folds
        'normalize': Categorical([True, False]),  # Whether to normalize the input data
    }
}


class RidgeCVHyperparameterTuner:
    _search_space: str
    _best_model: RidgeCV | None
    _best_params: dict | None
    _y_train: pd.Series | None
    _y_pred: pd.Series | None
    __name__ = 'RidgeCVRegressor'

    def __init__(self, search_space='default'):
        self._search_space = param_spaces[search_space] if search_space in param_spaces.keys() else param_spaces['default']

    def fit(self, X, y):
        np.int = int
        search_cv = BayesSearchCV(
            estimator=RidgeCV(),
            search_spaces=self._search_space,
            n_iter=30,
            scoring='neg_mean_squared_error',
            cv=5,
            n_jobs=-1,
            random_state=42
        )
        search_cv.fit(X=X, y=y)

        regressor = RidgeCV(**search_cv.best_params_)
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
