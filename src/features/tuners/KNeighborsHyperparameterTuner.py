import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.neighbors import KNeighborsRegressor
from skopt import BayesSearchCV
from skopt.space import Integer, Categorical
from sklearn.metrics import root_mean_squared_error, r2_score, PredictionErrorDisplay

param_spaces = {
    'default': {
        'n_neighbors': Integer(1, 30),  # Number of neighbors to consider, covering both small and large values
        'weights': Categorical(['uniform', 'distance']),  # Weighting of neighbors, with 'distance' giving closer neighbors more weight
        'p': Categorical([1, 2]),  # Power parameter for Minkowski distance, with 1 being Manhattan and 2 being Euclidean
        'leaf_size': Integer(10, 50),  # Leaf size for the tree-based neighbor search, affects efficiency
    },
}


class KNeighborsHyperparameterTuner:
    _search_space: str
    _best_model: KNeighborsRegressor | None
    _best_params: dict | None
    _y_train: pd.Series | None
    _y_pred: pd.Series | None
    __name__ = 'KNeighborsRegressor'

    def __init__(self, search_space='default'):
        self._search_space = param_spaces[search_space] if search_space in param_spaces.keys() else param_spaces['default']

    def fit(self, X, y):
        np.int = int
        regressor = KNeighborsRegressor()
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

        regressor = KNeighborsRegressor(**search_cv.best_params_)
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
