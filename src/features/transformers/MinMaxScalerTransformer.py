from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.preprocessing import MinMaxScaler


class MinMaxScalerTransformer(BaseEstimator, TransformerMixin):
    _columns: list[str] | None = None
    _types: list[str] | None = None
    _scaler: MinMaxScaler | None = None

    def __init__(self, feature_range=(0, 1), columns: list[str] | None = None):
        """
        Custom transformer that applies MinMax scaling to specified columns.

        Parameters:
        - feature_range: tuple (min, max), default=(0, 1)
          Desired range of transformed data.
        - columns: list, default=None
          Columns to be scaled. If None, all columns are scaled.
        """
        self._columns = columns
        self._scaler = MinMaxScaler(feature_range=feature_range)

    def fit(self, X, y=None):
        # Initialize the scaler with the specified range
        if self._columns is not None:
            self._scaler.fit(X[self._columns])
            return self

        return self

    def transform(self, X):
        # Make a copy of the DataFrame to avoid modifying the original data
        X_scaled = X.copy()

        # Apply the scaler to the specified columns and assign the transformed values back
        X_scaled[self._columns] = self._scaler.transform(X[self._columns])
        return X_scaled
