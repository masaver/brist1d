import os

import pandas as pd

from src.features.helpers.load_data import load_data as common_load_data
from src.models.model_2.model.pipelines import pipeline

filename_x_train = os.path.join(os.path.dirname(__file__), 'X_train.csv')
filename_y_train = os.path.join(os.path.dirname(__file__), 'y_train.csv')
filename_x_test = os.path.join(os.path.dirname(__file__), 'X_test.csv')


def load_data(recreate: bool = False) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    if os.path.exists(filename_x_train) and os.path.exists(filename_y_train) and os.path.exists(filename_x_test):
        if not recreate:
            X_train = pd.read_csv(filename_x_train, index_col=0)
            y_train = pd.read_csv(filename_y_train, index_col=0)['bg+1:00']
            X_test = pd.read_csv(filename_x_test, index_col=0)
            return X_train, y_train, X_test

    train_data, augmented_data, test_data = common_load_data('1_00h', True)

    train_data_transformed = pipeline.fit_transform(pd.concat([train_data, augmented_data]))
    test_data_transformed = pipeline.transform(test_data)

    X_train = train_data_transformed.drop(columns=['bg+1:00'])
    X_train.to_csv(filename_x_train)

    y_train = train_data_transformed['bg+1:00']
    y_train.to_csv(filename_y_train)

    X_test = test_data_transformed
    X_test.to_csv(filename_x_test)

    return X_train, y_train, X_test
