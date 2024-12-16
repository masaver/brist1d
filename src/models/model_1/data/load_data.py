import os

import pandas as pd
from sklearn.pipeline import Pipeline
import joblib

from src.features.helpers.load_data import load_data as common_load_data
from src.models.model_1.model.pipelines import pipeline

filename_x_train = os.path.join(os.path.dirname(__file__), 'X_train.csv')
filename_y_train = os.path.join(os.path.dirname(__file__), 'y_train.csv')
filename_x_test = os.path.join(os.path.dirname(__file__), 'X_test.csv')
data_pipeline_file = os.path.join(os.path.dirname(__file__), 'pipeline.pkl')
timestamps_file = os.path.join(os.path.dirname(__file__), 'timestamps.csv')

def load_data(recreate: bool = False) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, Pipeline, pd.DataFrame]:
    cond = os.path.exists(filename_x_train) and \
           os.path.exists(filename_y_train) and \
           os.path.exists(filename_x_test) and \
           os.path.exists(data_pipeline_file) and \
           os.path.exists(timestamps_file)
    
    if cond:
        if not recreate:
            X_train = pd.read_csv(filename_x_train, index_col=0)
            y_train = pd.read_csv(filename_y_train, index_col=0)['bg+1:00']
            X_test = pd.read_csv(filename_x_test, index_col=0)
            data_pipeline = joblib.load(data_pipeline_file)
            time_stamps_df = pd.read_csv(timestamps_file, index_col=0)
            return X_train, y_train, X_test, data_pipeline, time_stamps_df

    train_data, augmented_data, test_data = common_load_data('1_00h', True)

    # Transform the data and split features and target variables
    train_data_transformed = pipeline.fit_transform(train_data)
    X_train = train_data_transformed.drop(columns=['bg+1:00'])
    y_train = train_data['bg+1:00']
    X_test = pipeline.fit_transform(test_data)

    # Save a table the the time
    time_stamps_df = train_data['time']

    # Save the processed data
    X_train.to_csv(filename_x_train)
    y_train.to_csv(filename_y_train)
    X_test.to_csv(filename_x_test)
    joblib.dump(pipeline, data_pipeline_file)
    time_stamps_df.to_csv(timestamps_file)

    return X_train, y_train, X_test, pipeline, time_stamps_df
