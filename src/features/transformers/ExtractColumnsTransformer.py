import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class ExtractColumnsTransformer(BaseEstimator, TransformerMixin):
    _columns_to_extract: list[str]
    _starts_with: list[str] | str | None
    _ends_with: list[str] | str | None

    def __init__(self, columns_to_extract: list | None = None, starts_with: list[str] | str | None = None, ends_with: list[str] | str | None = None):
        self._columns_to_extract = columns_to_extract if columns_to_extract is not None else []
        self._starts_with = []
        if starts_with is not None:
            if type(starts_with) == str:
                self._starts_with = [starts_with]
            if type(starts_with) == list:
                self._starts_with = starts_with

        self._ends_with = []
        if ends_with is not None:
            if type(ends_with) == str:
                self._ends_with = [ends_with]
            if type(ends_with) == list:
                self._ends_with = ends_with

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def _get_columns_to_extract(self, X: pd.DataFrame):
        columns_to_extract = []
        for col in X.columns:
            if col in self._columns_to_extract:
                columns_to_extract.append(col)
                continue

            if any([col.startswith(starts_with) for starts_with in self._starts_with]):
                columns_to_extract.append(col)
                continue

            if any([col.endswith(ends_with) for ends_with in self._ends_with]):
                columns_to_extract.append(col)
                continue

        return columns_to_extract

    def transform(self, X: pd.DataFrame):
        columns_to_extract = self._get_columns_to_extract(X)
        return X[columns_to_extract]
