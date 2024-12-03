import numpy as np
import pandas as pd
import shap
from matplotlib import pyplot as plt
from sklearn.base import RegressorMixin
from sklearn.metrics import root_mean_squared_error, r2_score, PredictionErrorDisplay
from skopt import BayesSearchCV
from sklearn.model_selection import KFold, GridSearchCV, PredefinedSplit


class BaseHyperparameterTuner:
    _param_space: dict | None
    _best_model: RegressorMixin | None
    _best_params: dict | None
    _X_train: pd.DataFrame | None
    _y_train: pd.Series | None
    _y_pred_train: pd.Series | None

    _X_test: pd.DataFrame | None
    _y_test: pd.Series | None
    _y_pred_test: pd.Series | None

    _cv: KFold | PredefinedSplit | None
    __name__ = 'BaseHyperparameterTuner'

    @staticmethod
    def regressor():
        raise NotImplementedError

    @staticmethod
    def param_space(search_space: str | None) -> dict | None:
        raise NotImplementedError

    def __init__(self, search_space='default'):
        self._param_space = self.param_space(search_space)

    def fit(self, X_train, y_train, X_test=None, y_test=None):
        np.int = int
        self._X_train = X_train
        self._y_train = y_train
        self._y_pred_train = None
        self._X_test = X_test
        self._y_test = y_test
        self._y_pred_test = None

        self._cv = KFold(n_splits=5, shuffle=True, random_state=42)
        if X_test is not None and y_test is not None:
            self._cv = PredefinedSplit(test_fold=[-1] * len(X_train) + [0] * len(X_test))

        # default search cv
        search_cv = GridSearchCV(
            estimator=self.regressor(),
            param_grid={},
            scoring='neg_mean_squared_error',
            cv=self._cv,
            n_jobs=-1,
        )

        if self._param_space is not None:
            search_cv = BayesSearchCV(
                estimator=self.regressor(),
                search_spaces=self._param_space,
                n_iter=30,
                scoring='neg_mean_squared_error',
                cv=self._cv,
                n_jobs=-1,
                random_state=42
            )

        X = pd.concat([self._X_train, self._X_test]) if X_test is not None else self._X_train
        y = pd.concat([self._y_train, self._y_test]) if y_test is not None else self._y_train

        search_cv.fit(X=X, y=y)
        self._best_model = search_cv.best_estimator_
        self._best_params = search_cv.best_params_

        if self._X_test is not None and self._y_test is not None:
            self._y_pred_test = search_cv.predict(X=self._X_test)

        self._y_pred_train = search_cv.predict(X=self._X_train)

    def get_params(self):
        if self._best_params is None:
            raise ValueError('No model has been fitted yet')

        print(f'Best Parameters for {self.__name__}')
        return self._best_params

    def get_r2_score(self):
        if self._y_pred_test is not None and self._y_test is not None:
            return r2_score(y_true=self._y_test, y_pred=self._y_pred_test)

        if self._y_pred_train is not None and self._y_train is not None:
            return r2_score(y_true=self._y_train, y_pred=self._y_pred_train)

        raise ValueError('No model has been fitted yet')

    def get_rmse(self):
        if self._y_pred_test is not None and self._y_test is not None:
            return root_mean_squared_error(y_true=self._y_test, y_pred=self._y_pred_test)

        if self._y_pred_train is not None and self._y_train is not None:
            return root_mean_squared_error(y_true=self._y_train, y_pred=self._y_pred_train)

        raise ValueError('No model has been fitted yet')

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
        if self._y_train is not None and self._y_pred_train is not None:
            print('Train Set')
            self.result_summary(y_true=self._y_train, y_pred=self._y_pred_train)
        if self._y_test is not None and self._y_pred_test is not None:
            print('Test Set')
            self.result_summary(y_true=self._y_test, y_pred=self._y_pred_test)

    @staticmethod
    def result_summary(y_true, y_pred, title=None):
        if title is not None:
            print(title)
        print(f'RMSE: {root_mean_squared_error(y_true, y_pred)}')
        print(f'R2 Score: {r2_score(y_true, y_pred)}')
        fig, axs = plt.subplots(ncols=2, figsize=(8, 4))
        PredictionErrorDisplay.from_predictions(
            y_true=y_true,
            y_pred=y_pred,
            kind="actual_vs_predicted",
            subsample=100,
            ax=axs[0],
            random_state=0,
        )
        axs[0].set_title("Actual vs. Predicted values")
        PredictionErrorDisplay.from_predictions(
            y_true=y_true,
            y_pred=y_pred,
            kind="residual_vs_predicted",
            subsample=100,
            ax=axs[1],
            random_state=0,
        )
        axs[1].set_title("Residuals vs. Predicted Values")
        fig.suptitle("Plotting cross-validated predictions")
        plt.tight_layout()
        plt.show()
