import pandas as pd
import numpy as np
import re
from sklearn.base import BaseEstimator, TransformerMixin

class RenameColumnsTransformer(BaseEstimator, TransformerMixin):
    """
    Custom transformer to rename columns in a DataFrame using a single rule.
    """
    def __init__(self, rename_rule = 'custom'):
        """
        Parameters:
        - rename_rule: callable
            A function that takes a column name as input and returns the renamed column name.
        """
        self.rename_rule = rename_rule

    def fit(self, X, y=None):
        # No fitting necessary for column renaming
        return self

    def transform(self, X):
        """
        Apply the renaming rule to all columns.
        """
        if self.rename_rule == 'custom':
            Xs = X.copy()
            Xs.columns = [re.sub(r'[^a-zA-Z0-9_]', '_', col) for col in Xs.columns ]
            rename_dict = {'bg_1_00': 'bg+1:00'}
            Xs.rename( columns = rename_dict , inplace = True )
        
        return Xs
