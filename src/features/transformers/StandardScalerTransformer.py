import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler


class StandardScalerTransformer(BaseEstimator, TransformerMixin):
    _columns: list[str] | None = None
    _types: list[str] | None = None
    _scaler: StandardScaler | None = None

    def __init__(self, columns: list[str] | None = None, types: list[str] | None = None):
        """
        Initializes the ColumnScaler with the specified columns to scale.

        Parameters:
        columns (list): List of columns to apply StandardScaler to. If None, no scaling is applied.
        types (list, optional): List of data types of the columns. Defaults to None.
        """
        self._columns = columns 
        self._types = types if types is not None else [np.number]
        self._scaler = StandardScaler()

    def fit(self, X, y=None):
        """
        Fits the StandardScaler to the specified columns in the DataFrame.

        Parameters:
        X (DataFrame): Input DataFrame.
        y (Series, optional): Target variable. Defaults to None.

        Returns:
        self: Fitted transformer.
        """
        if self._columns is not None:
            self._scaler.fit(X[self._columns])
            return self

        ## If no columns are specified, fit all columns of the specified types, defaulting to all numeric columns
        self._scaler.fit(X.select_dtypes(include=self._types))
        return self

    def transform(self, X):
        """
        Transforms the specified columns using the fitted StandardScaler.

        Parameters:
        X (DataFrame): Input DataFrame.

        Returns:
        DataFrame: Transformed DataFrame with scaled specified columns.
        """
        X_copy = X.copy()
        if self._columns is not None:
            X_copy[self._columns] = self._scaler.transform(X[self._columns])
            return X_copy

        ## If no columns are specified, scale all columns of the specified types, defaulting to all numeric columns
        X_copy[X.select_dtypes(include=self._types).columns] = self._scaler.transform(X.select_dtypes(include=self._types))
        return X_copy