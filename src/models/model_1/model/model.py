import os

from sklearn.ensemble import VotingRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import Lasso, Ridge
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import joblib

MODEL_FILE_NAME = os.path.join(os.path.dirname(__file__), 'model.pkl')

# Best params from tuners
lasso_best_params = {'alpha':0.0009766639779349325}
ridge_best_params = {'alpha': 0.4678860424711244}

hist_best_params = {
    'early_stopping': False,
    'l2_regularization': 1.0,
    'learning_rate': 0.12054185358199056,
    'max_bins': 255,
    'max_depth': 15,
    'max_iter': 500,
    'max_leaf_nodes': 31,
    'min_samples_leaf': 10,
    'scoring': 'neg_mean_squared_error'
    }

xgb_best_params = {
    'alpha': 0.1,
    'colsample_bytree': 0.85,
    'gamma': 0,
    'lambda': 0.1,
    'learning_rate': 0.1,
    'max_depth': 7,
    'min_child_weight': 5,
    'n_estimators': 300,
    'subsample': 0.85
    }

lgbm_best_params = {
    'colsample_bytree': 1.0,
    'learning_rate': 0.04698836347404239,
    'max_depth': 15,
    'min_child_samples': 10,
    'min_child_weight': 4.309647054433731,
    'n_estimators': 974,
    'num_leaves': 150,
    'reg_alpha': 10.0,
    'reg_lambda': 1e-08,
    'subsample': 0.8815860111723086
    }

class Model():
    _model: VotingRegressor = None

    def __init__(self, load: bool = False):
        if load:
            if os.path.exists(MODEL_FILE_NAME) and os.path.isfile(MODEL_FILE_NAME) and os.path.getsize(MODEL_FILE_NAME) > 0:
                self._model = joblib.load(MODEL_FILE_NAME)
            else:
                print('Model file not found or empty. Creating a new model instance.')

        if self._model is None:
            self._model = VotingRegressor( 
                estimators=[
                    ('lasso', Lasso( **lasso_best_params )),
                    ('ridge', Ridge( **ridge_best_params )),
                    ('hist', HistGradientBoostingRegressor( **hist_best_params )),
                    ('xgb',XGBRegressor( **xgb_best_params )),
                    ('lgbm',LGBMRegressor( **lgbm_best_params ))
                ])

    def fit(self, X, y):
        self._model.fit(X, y)

    def predict(self, X):
        return self._model.predict(X)

    def save(self):
        joblib.dump(self._model, MODEL_FILE_NAME)
        pass

    def load(self):
        self._model = joblib.load(MODEL_FILE_NAME)
        pass

