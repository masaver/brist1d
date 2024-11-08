import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class GetDummiesTransformer(BaseEstimator, TransformerMixin):
    _columns: list[str]

    def __init__(self, columns=None):
        self._columns = columns if columns is not None else []

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        columns_to_apply = [col for col in self._columns if col in X.columns]
        if len(columns_to_apply) == 0:
            return X

        return pd.get_dummies(X, columns=self._columns, drop_first=True)
