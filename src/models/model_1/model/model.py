import os

from sklearn.ensemble import StackingRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import Ridge, LassoLarsIC
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
import joblib

MODEL_FILE_NAME = os.path.join(os.path.dirname(__file__), 'model.pkl')


class Model():
    _model: StackingRegressor = None

    def __init__(self, load: bool = False):
        if load:
            if os.path.exists(MODEL_FILE_NAME) and os.path.isfile(MODEL_FILE_NAME) and os.path.getsize(MODEL_FILE_NAME) > 0:
                self._model = joblib.load(MODEL_FILE_NAME)
            else:
                print('Model file not found or empty. Creating a new model instance.')

        if self._model is None:
            self._model = StackingRegressor(
                estimators=[
                    ('hgb', HistGradientBoostingRegressor(max_iter=200, max_depth=5, learning_rate=0.01)),
                    ('lasso', LassoLarsIC(criterion='bic', max_iter=10000)),
                    ('knn', KNeighborsRegressor(n_neighbors=5)),
                    ('xgb', XGBRegressor(objective='reg:squarederror', random_state=42, n_estimators=500, max_depth=5, learning_rate=0.1)),
                ],
                final_estimator=Ridge(alpha=0.1), n_jobs=1, verbose=2
            )

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
