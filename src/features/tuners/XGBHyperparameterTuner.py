import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from skopt import BayesSearchCV
from xgboost import XGBRegressor
from skopt.space import Integer, Real
import shap
from sklearn.metrics import root_mean_squared_error, r2_score, PredictionErrorDisplay

param_spaces = {
    'default': {
        'n_estimators': Integer(100, 300),  # Common range for boosting rounds
        'learning_rate': Real(0.03, 0.1, 'log-uniform'),  # Standard learning rates
        'max_depth': Integer(3, 7),  # Default depth range that balances depth and overfitting
        'min_child_weight': Integer(1, 5),  # Moderate range to allow small variations in regularization
        'subsample': Real(0.7, 0.85),  # Light subsampling for diversity while preserving data
        'colsample_bytree': Real(0.7, 0.85),  # Common column sampling range
        'gamma': Integer(0, 3),  # Small gamma range for light regularization
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


class XGBHyperparameterTuner:
    _search_space: str
    _best_model: XGBRegressor | None
    _best_params: dict | None
    _y_train: pd.Series | None
    _y_pred: pd.Series | None
    __name__ = 'XGBRegressor'

    def __init__(self,  n = None , search_space = 'default' ):
        self._n = n
        self._search_space = param_spaces[search_space] if search_space in param_spaces.keys() else param_spaces['default']

    def fit(self, X, y):
        # np.int = int
        
        # Fit athe base model
        estimator = XGBRegressor(objective='reg:squarederror', random_state=42, tree_method='hist')
        estimator.fit(X,y)

        # # Do a SHAP analysis
        explainer = shap.Explainer( estimator )
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

        #
        search_cv = BayesSearchCV(
            estimator=XGBRegressor(objective='reg:squarederror', random_state=42, tree_method='hist'),
            search_spaces=self._search_space,
            n_iter=30,
            scoring='neg_mean_squared_error',
            cv=5,
            n_jobs=-1,
            random_state=42
        )
        
        if self._n is not None:
            search_cv.fit( X = X[self.top_n_features] , y = y )
            regressor = XGBRegressor(**search_cv.best_params_, objective='reg:squarederror', random_state=42, tree_method='hist')
            regressor.fit(X=X[self.top_n_features], y=y)
            self._y_pred = regressor.predict(X=X[self.top_n_features])

        
        if self._n is None:
            search_cv.fit( X = X , y = y )
            regressor = XGBRegressor(**search_cv.best_params_, objective='reg:squarederror', random_state=42, tree_method='hist')
            regressor.fit(X=X, y=y)
            self._y_pred = regressor.predict(X=X)

        self._best_model = regressor
        self._best_params = search_cv.best_params_
        self._y_train = y
        
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
    
    def get_best_features(self):
        if self._n is None:
            raise ValueError('No features selected based on SHAP')
        return self.top_n_features

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
        
        if self._n is not None:
            shap.plots.beeswarm( self._shap_values[:,self.top_n_features] , max_display = len( self.top_n_features ) )
        
        if self._n is None:
            shap.plots.beeswarm( self._shap_values[:,self.top_n_features] )
