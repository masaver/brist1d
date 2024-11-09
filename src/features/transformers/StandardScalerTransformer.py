from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
import pandas as pd

class StandardScalerTransformer(BaseEstimator, TransformerMixin):
    _columns = None
    _scaler = None

    def __init__(self, columns=None):
        """
        Initializes the ColumnScaler with the specified columns to scale.

        Parameters:
        columns (list): List of columns to apply StandardScaler to. 
        
        MS mod: If  collumns = None, assumes all columns to be numeric, and Standarizes them
        """
        self._columns = columns
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
        if self._columns is None:
             self._scaler.fit(X)

        if self._columns is not None:
            self._scaler.fit(X[self._columns])

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

        if self._columns is None:
             X_copy = pd.DataFrame( self._scaler.transform( X_copy ) , index = X_copy.index , columns = X_copy.columns )

        if self._columns is not None:
            X_copy[self._columns] = self._scaler.transform(X[self._columns])

        return X_copy
