from typing import Literal

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class DateTimeHourTransformer(BaseEstimator, TransformerMixin):
    _result_columns: list[str]
    _source_time_format: str
    _time_column: str
    _type: Literal['sin_cos', 'bins']

    def __init__(self, time_column: str, result_column: str, type: Literal['sin_cos', 'bins'] = 'bins', drop_time_column: bool = False, number_of_bins: int = 24,
                 source_time_format: str = '%H:%M:%S'):
        self._time_column = time_column
        self._drop_time_column = drop_time_column
        self._result_column = result_column
        self._type = type
        self._number_of_bins = number_of_bins
        self._source_time_format = source_time_format

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame):
        X = X.copy()

        if not self._time_column in X.columns:
            raise ValueError('time_column must be set')

        if self._type == 'sin_cos':
            hour_values = pd.to_datetime(X[self._time_column], format=self._source_time_format).dt.hour
            minute_values = pd.to_datetime(X[self._time_column], format=self._source_time_format).dt.minute

            X[f'{self._result_column}_sin'] = np.sin(2 * np.pi * (hour_values + minute_values / 60) / 24)
            X[f'{self._result_column}_cos'] = np.cos(2 * np.pi * (hour_values + minute_values / 60) / 24)
            columns = list(X.columns)
            columns.remove(f'{self._result_column}_sin')
            columns.remove(f'{self._result_column}_cos')
            columns.insert(columns.index(self._time_column) + 1, f'{self._result_column}_sin')
            columns.insert(columns.index(self._time_column) + 2, f'{self._result_column}_cos')
            X = X[columns]

        elif self._type == 'bins':
            time_bins = np.linspace(0, 24, self._number_of_bins + 1)
            hour_values = pd.to_datetime(X[self._time_column], format=self._source_time_format).dt.hour
            X[self._result_column] = pd.cut(hour_values, bins=time_bins, labels=range(self._number_of_bins))
            columns = list(X.columns)
            columns.remove(self._result_column)
            columns.insert(columns.index(self._time_column) + 1, self._result_column)
            X = X[columns]

        if self._drop_time_column:
            X = X.drop(columns=[self._time_column])

        return X
