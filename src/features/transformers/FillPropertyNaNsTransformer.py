import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from typing import Literal

parameters = ['bg', 'insulin', 'carbs', 'hr', 'steps', 'cals', 'activity']
time_diffs = [
    '-0:00',
    '-0:05',
    '-0:10',
    '-0:15',
    '-0:20',
    '-0:25',
    '-0:30',
    '-0:35',
    '-0:40',
    '-0:45',
    '-0:50',
    '-0:55',
    '-1:00',
    '-1:05',
    '-1:10',
    '-1:15',
    '-1:20',
    '-1:25',
    '-1:30',
    '-1:35',
    '-1:40',
    '-1:45',
    '-1:50',
    '-1:55',
    '-2:00',
    '-2:05',
    '-2:10',
    '-2:15',
    '-2:20',
    '-2:25',
    '-2:30',
    '-2:35',
    '-2:40',
    '-2:45',
    '-2:50',
    '-2:55',
    '-3:00',
    '-3:05',
    '-3:10',
    '-3:15',
    '-3:20',
    '-3:25',
    '-3:30',
    '-3:35',
    '-3:40',
    '-3:45',
    '-3:50',
    '-3:55',
    '-4:00',
    '-4:05',
    '-4:10',
    '-4:15',
    '-4:20',
    '-4:25',
    '-4:30',
    '-4:35',
    '-4:40',
    '-4:45',
    '-4:50',
    '-4:55',
    '-5:00',
    '-5:05',
    '-5:10',
    '-5:15',
    '-5:20',
    '-5:25',
    '-5:30',
    '-5:35',
    '-5:40',
    '-5:45',
    '-5:50',
    '-5:55'
]

Parameter = Literal['bg', 'insulin', 'carbs', 'hr', 'steps', 'cals', 'activity']
How = Literal['mean', 'median', 'zero', 'interpolate']


class FillPropertyNaNsTransformer(BaseEstimator, TransformerMixin):
    _parameter: Parameter | str
    _hows: list[How] | list[str]
    _mean: None | float
    _median: None | float
    _precision: None | int
    _ffill: bool | int
    _bfill: bool | int
    _interpolate: bool | int

    def __init__(self,
                 parameter: Parameter | str,
                 how: How | list[How] | str | list[str] = 'median',
                 precision: int | None = None,
                 ffill: bool | int = True,
                 bfill: bool | int = True,
                 interpolate: bool | int = True):
        if not parameter in parameters:
            raise ValueError(f'parameter must be one of {parameters}')

        hows = how if isinstance(how, list) else [how]
        for how in hows:
            if not how in ['mean', 'median', 'zero', 'interpolate']:
                raise ValueError(f'how must be one of mean, median, zero, interpolate')

        self._parameter = parameter
        self._hows = hows
        self._precision = precision
        self._ffill = ffill
        self._bfill = bfill
        self._interpolate = interpolate

    def fit(self, X: pd.DataFrame, y=None):
        self._mean = X[self._get_affection_columns(X=X)].stack().mean(numeric_only=True)
        self._median = X[self._get_affection_columns(X=X)].stack().median(numeric_only=True)
        return self

    def _get_affection_columns(self, X: pd.DataFrame):
        columns = X.columns
        affected_columns = []
        for time_diff in time_diffs:
            affected_column = f'{self._parameter}{time_diff}'
            if affected_column in columns:
                affected_columns.append(affected_column)

        return affected_columns

    def transform(self, X: pd.DataFrame):
        X = X.copy()
        columns = self._get_affection_columns(X)
        for how in self._hows:
            if how == 'mean':
                X[columns] = X[columns].fillna(self._mean)

            if how == 'median':
                X[columns] = X[columns].fillna(self._median)

            if how == 'zero':
                X[columns] = X[columns].fillna(0)

            if how == 'interpolate':
                limit = self._interpolate if type(self._interpolate) == int else None
                X[columns] = X[columns].interpolate(axis=1, limit=limit)

                if self._ffill:
                    limit = self._ffill if type(self._ffill) == int else None
                    X[columns] = X[columns].ffill(axis=1, limit=limit)

                if self._bfill:
                    limit = self._bfill if type(self._bfill) == int else None
                    X[columns] = X[columns].bfill(axis=1, limit=limit)

                if self._precision:
                    X[columns] = X[columns].round(self._precision)

        return X
