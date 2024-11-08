import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class DropColumnsTransformer(BaseEstimator, TransformerMixin):
    _columns_to_delete: list[str]
    _starts_with: list[str] | str | None

    def __init__(self, columns_to_delete: list | None = None, starts_with: list[str] | str | None = None):
        self._columns_to_delete = columns_to_delete if columns_to_delete is not None else []
        self._starts_with = []

        if starts_with is not None:
            if type(starts_with) == str:
                self._starts_with = [starts_with]
            if type(starts_with) == list:
                self._starts_with = starts_with

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def _get_columns_to_delete(self, X: pd.DataFrame):
        columns_to_delete = []
        for col in X.columns:
            if col in self._columns_to_delete:
                columns_to_delete.append(col)
                continue
            if any([col.startswith(starts_with) for starts_with in self._starts_with]):
                columns_to_delete.append(col)
                continue

        return columns_to_delete

    def transform(self, X: pd.DataFrame):
        columns_to_delete = self._get_columns_to_delete(X)
        return X.drop(columns=columns_to_delete)
