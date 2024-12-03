import os

import pandas as pd

from src.features.helpers.load_data import load_data as common_load_data
from src.models.model_1.model.pipelines import pipeline

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

    # Transform the data and split feaures and target variables
    train_data_transformed = pipeline.fit_transform( train_data )
    X_train = train_data_transformed.drop( 'bg+1:00' , axis = 1 )
    y_train = train_data['bg+1:00']
    X_test = pipeline.fit_transform(test_data)

    # fix column names - Needed for LGBM
    import re
    X_train.columns = [re.sub(r'[^a-zA-Z0-9_]', '_', col) for col in X_train.columns]
    X_test.columns = [re.sub(r'[^a-zA-Z0-9_]', '_', col) for col in X_test.columns]

    #Subset  to keep top features opnly
    top_feat = ['hr_0_00', 'bg_0_15', 'day_phase_evening', 'bg_0_00', 'insulin_0_00', 'day_phase_night', 'bg_0_10']
    X_train = X_train[ top_feat ]
    X_test = X_test[ top_feat ]

    # Save the processed data
    X_train.to_csv(filename_x_train)
    y_train.to_csv(filename_y_train)
    X_test.to_csv(filename_x_test)

    return X_train , y_train , X_test
