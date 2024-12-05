# Transformer classes

## Transforming Date and Time Features

## DateTimeHourTransformer

The `DateTimeHourTransformer` has two different modes of operation.

* `sin_cos`, which converts the hour of the day into two features, `hour_sin` and `hour_cos`, which represent the hour of the day as a sine and cosine wave.
* `bins`, which converts the hour of the day into a categorical feature with a specified number of bins

```python
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

```

## DayPhaseTransformer

The `DayPhaseTransformer` converts the hour of the day into a categorical feature representing 6 phases of the day:

* `morning` - 6 to 9
* `noon` - 10 to 13
* `afternoon` - 14 to 17
* `evening` - 18 to 21
* `late_evening` - 22 to 24
* `night` - 0 to 5

```python
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class DayPhaseTransformer(BaseEstimator, TransformerMixin):
  _result_column: str
  _time_column: str

  def __init__(self, time_column: str, time_format: str, result_column: str, drop_time_column: bool = False, ignore_errors: bool = False):
    self._time_column = time_column
    self._time_format = time_format
    self._result_column = result_column
    self._drop_time_column = drop_time_column
    self._ignore_errors = ignore_errors

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
    X = X.copy()
    if not self._time_column in X.columns:
      if self._ignore_errors:
        return X
      raise ValueError('time_column must be set')

    X[self._result_column] = pd.to_datetime(X[self._time_column], format=self._time_format).dt.hour.apply(self._get_day_phase)

    # reorder result column directly after time column
    columns = list(X.columns)
    columns.remove(self._result_column)
    columns.insert(columns.index(self._time_column) + 1, self._result_column)
    X = X[columns]

    if self._drop_time_column:
      X = X.drop(columns=[self._time_column])

    return X
```

## DropColumnsTransformer

The `DropColumnsTransformer` removes columns from a DataFrame based on a list of column names or columns that start with a specified string. This makes it easy to remove all
columns from one property, for example.

```python
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
```

### FillPropertyNaNsTransformer

The `FillPropertyNaNsTransformer` fills NaN values in a DataFrame based on the specified parameter and fill strategy. The transformer supports the following fill strategies:

* `mean` - Fill NaN values with the mean of the column
* `median` - Fill NaN values with the median of the column
* `zero` - Fill NaN values with zero
* `interpolate` - Fill NaN values by interpolating between the nearest values with optional limit
* `ffill` - Fill NaN values by forward filling with optional limit
* `bfill` - Fill NaN values by backward filling with optional limit

The methods can be passed as a list to apply multiple fill strategies in sequence.

```python
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from typing import Literal

parameters = ['bg', 'insulin', 'carbs', 'hr', 'steps', 'cals', 'activity']
time_diffs = [f'-{i}:{j:02}' for i in range(6) for j in range(0, 60, 5)]

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

```

## PropertyOutlierTransformer

The `PropertyOutlierTransformer` removes outliers from a DataFrame based on a specified parameter and filter function. The transformer supports the following fill strategies:

* `min` - Fill NaN values with the minimum of the column
* `max` - Fill NaN values with the maximum of the column
* `mean` - Fill NaN values with the mean of the column
* `median` - Fill NaN values with the median of the column
* `zero` - Fill NaN values with 0

```python
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
```
