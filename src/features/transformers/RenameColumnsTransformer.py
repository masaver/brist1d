import pandas as pd
import re
from sklearn.base import BaseEstimator, TransformerMixin

class RenameColumnsTransformer(BaseEstimator,TransformerMixin):
    def __init__(self,rename_mapping=None):
        self.rename_mapping = rename_mapping or {}

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if self.rename_mapping is None:
            X.columns = [re.sub(r'[^a-zA-Z0-9_]', '_', col) for col in X.columns]
        else:
            X = X.rename(columns=self.rename_mapping)
        return X
