import json
import os
from typing import Literal

import pandas as pd

root_folder = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
with open(os.path.join(root_folder, 'settings.json')) as f:
    settings = json.load(f)

CLEAN_DATA_DIR = str(os.path.join(root_folder, settings['CLEAN_DATA_DIR']))

AvailableTimeFrame = Literal['1_00h', '2_00h', '3_00h', '4_00h', '4_30h', '4_55h']
available_time_frames = ['1_00h', '2_00h', '3_00h', '4_00h', '4_30h', '4_55h']


def load_data(time_frame: AvailableTimeFrame, only_patients_in_test: bool = True) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if time_frame not in available_time_frames:
        raise ValueError(f'Invalid time frame: {time_frame}. Available time frames: {available_time_frames}')

    train_data_file = os.path.join(CLEAN_DATA_DIR, 'train.csv')
    train_data = pd.read_csv(train_data_file, index_col=0, low_memory=False)

    augmented_data_file = os.path.join(CLEAN_DATA_DIR, f'augmented_{time_frame}.csv')
    augmented_data = pd.read_csv(augmented_data_file, index_col=0, low_memory=False)

    test_data_file = os.path.join(CLEAN_DATA_DIR, 'test.csv')
    test_data = pd.read_csv(test_data_file, index_col=0, low_memory=False)

    if only_patients_in_test:
        unique_patients = test_data['p_num'].unique()
        train_data = train_data[train_data['p_num'].isin(unique_patients)]
        augmented_data = augmented_data[augmented_data['p_num'].isin(unique_patients)]
        test_data = test_data[test_data['p_num'].isin(unique_patients)]

    return train_data, augmented_data, test_data


def load_train_data() -> pd.DataFrame:
    train_data_file = os.path.join(CLEAN_DATA_DIR, 'train.csv')
    train_data = pd.read_csv(train_data_file, index_col=0, low_memory=False)

    return train_data


def load_test_data() -> pd.DataFrame:
    test_data_file = os.path.join(CLEAN_DATA_DIR, 'test.csv')
    test_data = pd.read_csv(test_data_file, index_col=0, low_memory=False)

    return test_data
