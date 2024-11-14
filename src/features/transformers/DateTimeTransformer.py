import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class DateTimeTransformer(BaseEstimator, TransformerMixin):
    _target_column: str
    _source_column: str
    _source_time_format: str
    _target_time_format: str
    _drop_time_column: bool

    def __init__(self, source_column: str, source_time_format: str, target_column: str, target_time_format: str, drop_time_column: bool = False):
        self._source_column = source_column
        self._source_time_format = source_time_format
        self._target_column = target_column
        self._target_time_format = target_time_format
        self._drop_time_column = drop_time_column

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame):
        X = X.copy()
        if not self._source_column in X.columns:
            raise ValueError('source_column must be set')

        X[self._target_column] = pd.to_datetime(X[self._source_column], format=self._source_time_format).dt.strftime(self._target_time_format)

        return X
