import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import r2_score, root_mean_squared_error, PredictionErrorDisplay
from sklearn.ensemble import ExtraTreesRegressor
from skopt import BayesSearchCV
from skopt.space import Integer, Real, Categorical
import shap

param_spaces = {

    'default': {
        'max_depth': Integer(1, 20),  # Depth of the tree, controlling complexity and overfitting
        'min_samples_split': Integer(2, 20),  # Minimum samples required to split an internal node
        'min_samples_leaf': Integer(1, 10),  # Minimum samples required in a leaf node
        'max_features': Real(0.1, 1.0),  # Fraction of features to consider at each split
    },

    'deep': {
        'n_estimators': Integer(100, 1000),                 # Wider range for number of trees
        'max_depth': Integer(5, 50),                        # Deeper trees if desired
        'min_samples_split': Integer(2, 50),                # Higher minimum split values
        'min_samples_leaf': Integer(1, 50),                 # Higher range for min samples at leaf
        'max_features': Categorical(['sqrt', 'log2', None]),  # Standard feature options
        'bootstrap': Categorical([True, False]),            # Bootstrap sampling option
        'max_leaf_nodes': Integer(10, 1000),                # Maximum number of leaf nodes per tree
        'min_impurity_decrease': Real(0.0, 0.5, 'uniform'), # Minimum impurity decrease for splitting
        'ccp_alpha': Real(0.0, 0.1, 'uniform')              # Complexity parameter for Minimal Cost-Complexity Pruning
    }

}


class ExtraTreesHyperparameterTuner:
    _search_space: str
    _best_model: ExtraTreesRegressor | None
    _best_params: dict | None
    _y_train: pd.Series | None
    _y_pred: pd.Series | None
    _n: int | None
    __name__ = 'ExtraTreesRegressor'

    def __init__(self, n = 15 , search_space='default' ):
        self._n = n
        self._search_space = param_spaces[search_space] if search_space in param_spaces.keys() else param_spaces['default']

    def fit(self, X, y):
        np.int = int

        #F
        search_cv = BayesSearchCV(
            estimator=ExtraTreesRegressor(random_state=42),
            search_spaces=self._search_space,
            n_iter=30,
            scoring='neg_mean_squared_error',
            cv=5,
            n_jobs=-1,
            random_state=42
        )

        search_cv.fit(X=X, y=y)
        regressor = ExtraTreesRegressor(**search_cv.best_params_, random_state=42)
        regressor.fit(X=X, y=y)

        # Do a SHAP analysis
        explainer = shap.Explainer( self._best_model )
        shap_values = explainer( X )
        self._shap_values = shap_values

        # Get the Top n best features
        mean_abs_shap_values = np.abs( self._shap_values.values ).mean(axis=0)
        feature_importance_df = pd.DataFrame({
            'feature': X.columns, 
            'mean_abs_shap_value': mean_abs_shap_values
        })

        feature_importance_df = feature_importance_df.sort_values(by='mean_abs_shap_value', ascending=False)
        
        # Get the top-N features (for example, top 10 features)
        if self._n is not None:
            feature_importance_df = feature_importance_df.head( self._n )

        self.top_n_features = list( feature_importance_df['feature'] )


        # update self
        self._best_model = regressor
        self._best_params = search_cv.best_params_
        self._X = X
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

    def show_shap(self):
        if self._best_model is None:
            raise ValueError('No model has been fitted yet')
        
        shap.plots.beeswarm( self._shap_values[:,self.top_n_features] , max_display = len( self.top_n_features ) )

    