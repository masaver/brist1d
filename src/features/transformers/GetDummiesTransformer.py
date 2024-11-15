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
        X[columns_to_apply] = X[columns_to_apply].astype(str)
        X_dummies = pd.get_dummies(X[columns_to_apply], drop_first=True, dtype=int)
        X = X.drop(columns=columns_to_apply)
        X = pd.concat([X, X_dummies], axis=1)

        return X
