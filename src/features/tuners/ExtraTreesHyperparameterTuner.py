import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import r2_score, root_mean_squared_error, PredictionErrorDisplay
from sklearn.tree import ExtraTreeRegressor
from skopt import BayesSearchCV

param_spaces = {
    'default': {
        'max_depth': (1, 20),  # Depth of the tree, controlling complexity and overfitting
        'min_samples_split': (2, 20),  # Minimum samples required to split an internal node
        'min_samples_leaf': (1, 10),  # Minimum samples required in a leaf node
        'max_features': (0.1, 1.0),  # Fraction of features to consider at each split
        'splitter': ['random', 'best'],  # Choose between best or random split at each node
    }
}


class ExtraTreesHyperparameterTuner:
    _search_space: str
    _best_model: ExtraTreeRegressor | None
    _best_params: dict | None
    _y_train: pd.Series | None
    _y_pred: pd.Series | None
    __name__ = 'ExtraTreeRegressor'

    def __init__(self, search_space='default'):
        self._search_space = param_spaces[search_space] if search_space in param_spaces.keys() else param_spaces['default']

    def fit(self, X, y):
        np.int = int
        search_cv = BayesSearchCV(
            estimator=ExtraTreeRegressor(random_state=42),
            search_spaces=self._search_space,
            n_iter=30,
            scoring='neg_mean_squared_error',
            cv=5,
            n_jobs=-1,
            random_state=42
        )

        search_cv.fit(X=X, y=y)
        regressor = ExtraTreeRegressor(**search_cv.best_params_, random_state=42)
        regressor.fit(X=X, y=y)
        self._best_model = regressor
        self._best_params = search_cv.best_params_
        self._y_train = y
        self._y_pred = regressor.predict(X)

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
