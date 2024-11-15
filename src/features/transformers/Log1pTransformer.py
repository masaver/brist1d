import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler


class Log1pTransformer(BaseEstimator, TransformerMixin):
    _columns: list[str] | None = None
    _dtypes: list[str] | None = None

    def __init__(self, columns: list[str] | None = None, dtypes: list[str] | None = None):
        self._columns = columns
        self._dtypes = dtypes if dtypes is not None else [np.number]

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_copy = X.copy()
        if self._columns is not None:
            X_copy[self._columns] = np.log1p(X[self._columns])
            return X_copy

        X_copy[X.select_dtypes(include=self._dtypes).columns] = np.log1p(X.select_dtypes(include=self._dtypes))
        return X_copy
