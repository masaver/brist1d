import numpy as np
import pandas as pd
import shap
from matplotlib import pyplot as plt
from sklearn.base import RegressorMixin
from sklearn.metrics import root_mean_squared_error, r2_score, PredictionErrorDisplay
from skopt import BayesSearchCV
from sklearn.model_selection import KFold, GridSearchCV


class BaseHyperparameterTuner:
    _param_space: dict | None
    _best_model: RegressorMixin | None
    _best_params: dict | None
    _X_train: pd.DataFrame | None
    _y_train: pd.Series | None
    _y_pred: pd.Series | None
    __name__ = 'BaseHyperparameterTuner'

    @staticmethod
    def regressor():
        raise NotImplementedError

    @staticmethod
    def param_space(search_space: str | None) -> dict | None:
        raise NotImplementedError

    def __init__(self, search_space='default'):
        self._param_space = self.param_space(search_space)

    def fit(self, X, y):
        np.int = int
        self._X_train = X
        self._y_train = y

        # when the search space is not defined use the default model without hyperparameter tuning
        cv = KFold(n_splits=5, shuffle=True, random_state=42)
        if self._param_space is None:
            search_cv = GridSearchCV(
                estimator=self.regressor(),
                param_grid={},
                scoring='neg_mean_squared_error',
                cv=cv,
                n_jobs=-1,
            )
            search_cv.fit(X=X, y=y)
            self._best_model = search_cv.best_estimator_
            self._best_params = search_cv.best_params_
            self._y_pred = search_cv.predict(X=X)
            return

        cv = KFold(n_splits=5, shuffle=True, random_state=42)
        search_cv = BayesSearchCV(
            estimator=self.regressor(),
            search_spaces=self._param_space,
            n_iter=30,
            scoring='neg_mean_squared_error',
            cv=cv,
            n_jobs=-1,
            random_state=42
        )
        search_cv.fit(X=X, y=y)

        self._best_model = search_cv.best_estimator_
        self._best_params = search_cv.best_params_
        self._y_pred = search_cv.predict(X=X)

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

    def get_feature_importance(self):
        if self._best_model is None:
            raise ValueError('No model has been fitted yet')

        explainer = shap.Explainer(self._best_model)
        shap_values = explainer(self._X_train)
        feature_importance_df = pd.DataFrame(list(zip(self._X_train.columns, np.abs(shap_values.values).mean(0))), columns=['feature', 'shap_values'])
        feature_importance_df = feature_importance_df.sort_values(by='shap_values', ascending=False)
        return feature_importance_df

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
