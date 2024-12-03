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

    # Split feature and target variables
    xtrain = train_data.drop( 'bg+1:00' , axis = 1 )
    ytrain = train_data['bg+1:00']

    # Apply preprocessing pipeline
    data_pipe = pipeline
    Xs = data_pipe.fit_transform( xtrain )
    Xs_test = data_pipe.transform( test_data )

    # fix column names - Needed for LGBM
    import re
    Xs.columns = [re.sub(r'[^a-zA-Z0-9_]', '_', col) for col in Xs.columns]
    Xs_test.columns = [re.sub(r'[^a-zA-Z0-9_]', '_', col) for col in Xs_test.columns]

    #Subset  to keep top features opnly
    top_feat = ['hr_0_00', 'bg_0_15', 'day_phase_evening', 'bg_0_00', 'insulin_0_00', 'day_phase_night', 'bg_0_10']
    Xs = Xs[ top_feat ]
    Xs_test = Xs_test[ top_feat ]

    # Save the processed data
    Xs.to_csv(filename_x_train)
    ytrain.to_csv(filename_y_train)
    Xs_test.to_csv(filename_x_test)

    return Xs, ytrain, Xs_test