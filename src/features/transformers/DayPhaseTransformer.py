import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class DayPhaseTransformer(BaseEstimator, TransformerMixin):
    _result_column: str
    _time_column: str

    def __init__(self, time_column: str, time_format: str, result_column: str, drop_time_column: bool = False):
        self._time_column = time_column
        self._time_format = time_format
        self._result_column = result_column
        self._drop_time_column = drop_time_column

    def _get_day_phase(self, hour: int):
        if 6 <= hour <= 9:
            return 'morning'
        elif 10 <= hour <= 13:
            return 'noon'
        elif 14 <= hour <= 17:
            return 'afternoon'
        elif 18 <= hour <= 21:
            return 'evening'
        elif 22 <= hour <= 24:
            return 'late_evening'
        else:
            return 'night'

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame):
        X[self._result_column] = pd.to_datetime(X[self._time_column], format=self._time_format).dt.hour.apply(self._get_day_phase)

        # reorder result column directly after time column
        columns = list(X.columns)
        columns.remove(self._result_column)
        columns.insert(columns.index(self._time_column) + 1, self._result_column)
        X = X[columns]

        if self._drop_time_column:
            X = X.drop(columns=[self._time_column])

        return X
