import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from typing import Literal

parameters = ['bg', 'insulin', 'carbs', 'hr', 'steps', 'cals', 'activity']
time_diffs = [f'-{i}:{j:02}' for i in range(6) for j in range(0, 60, 5)]

Parameter = Literal['bg', 'insulin', 'carbs', 'hr', 'steps', 'cals', 'activity']


class RollingAverageTransformer(BaseEstimator, TransformerMixin):
    _parameter: Parameter | str
    _window: int

    def __init__(self, parameter: Parameter | str, window: int = 3):
        if not parameter in parameters:
            raise ValueError(f'parameter must be one of {parameters}')

        self._parameter = parameter
        self._window = window

    def fit(self, X: pd.DataFrame, y=None):
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
        # Rolling average will be calculated for each row, so we need to transpose the dataframe to calculate the rolling average for each column
        # And then transpose it back to the original
        X[columns] = X[columns].T.rolling(window=self._window, min_periods=1).mean().T

        return X
