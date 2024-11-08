import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from typing import Literal, Callable

parameters = ['bg', 'insulin', 'carbs', 'hr', 'steps', 'cals', 'activity']

Parameter = Literal['bg', 'insulin', 'carbs', 'hr', 'steps', 'cals', 'activity']
FillStrategy = Literal['min', 'max', 'mean', 'median', 'zero']


class PropertyOutlierTransformer(BaseEstimator, TransformerMixin):
    _parameter: Parameter
    _fill_strategy: FillStrategy
    _filter_function: Callable

    def __init__(self, parameter: Parameter, filter_function: Callable, fill_strategy: FillStrategy = 'zero'):
        if not parameter in parameters:
            raise ValueError(f'parameter must be one of {parameters}')

        self._parameter = parameter
        self._filter_function = filter_function
        self._fill_strategy = fill_strategy

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def _get_affection_columns(self, X: pd.DataFrame):
        affected_columns = [col for col in X.columns if self._parameter in col]
        return affected_columns

    def _fill_outlier(self, x, X: pd.DataFrame, column: str):
        if self._fill_strategy == 'zero':
            return 0
        if self._fill_strategy == 'min':
            return x if x is not None else X[column].min()
        if self._fill_strategy == 'max':
            return x if x is not None else X[column].max()
        if self._fill_strategy == 'mean':
            return x if x is not None else X[column].mean()
        if self._fill_strategy == 'median':
            return x if x is not None else X[column].median()

    def transform(self, X: pd.DataFrame):
        X = X.copy()
        columns = self._get_affection_columns(X)
        for column in columns:
            X[column] = X[column].apply(lambda x: self._fill_outlier(x, X, column) if self._filter_function(x) else x)

        return X
