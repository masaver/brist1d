from datetime import datetime
import json

import numpy as np
from os.path import dirname
import os
import pandas as pd

from helpers.extractors import extract_patient_data, parameters, time_diffs, parse_time_diff
from transformers import FillPropertyNaNsTransformer


def get_time() -> str:
    return datetime.now().strftime("%H:%M:%S")


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


script_directory = dirname(os.path.abspath(__file__))
root_directory = dirname(dirname(dirname(__file__)))
print(f'{bcolors.OKGREEN}{get_time()} - Root directory: {root_directory}{bcolors.ENDC}')

with open(os.path.join(root_directory, 'settings.json')) as f:
    settings = json.load(f)

RAW_DATA_DIR = str(os.path.join(root_directory, settings['RAW_DATA_DIR']))
CLEAN_DATA_DIR = str(os.path.join(root_directory, settings['CLEAN_DATA_DIR']))

print(f'{bcolors.OKGREEN}{get_time()} - Raw data directory: {RAW_DATA_DIR}{bcolors.ENDC}')
print(f'{bcolors.OKGREEN}{get_time()} - Clean data directory: {CLEAN_DATA_DIR}{bcolors.ENDC}')

if __name__ == '__main__':
    print(f'{bcolors.OKGREEN}{get_time()} - Extracting patient data{bcolors.ENDC}')
    print(f'{bcolors.OKGREEN}{"=" * 50}{bcolors.ENDC}')
    print(f'{bcolors.OKCYAN}{get_time()} - Loading test.csv{bcolors.ENDC}')

    df = pd.read_csv(os.path.join(RAW_DATA_DIR, 'test.csv'), na_values=np.nan, low_memory=False)

    # fill bg NaN values with median
    df = FillPropertyNaNsTransformer(parameter='bg', how=['interpolate', 'median'], precision=2, interpolate=3, ffill=1,
                                     bfill=1).fit_transform(df)

    # fill insulin NaN values with median
    df = FillPropertyNaNsTransformer(parameter='insulin', how=['interpolate', 'median'], precision=2, interpolate=3,
                                     ffill=1, bfill=1).fit_transform(df)

    patients = df['p_num'].unique()
    print(f'{bcolors.OKCYAN}{get_time()} - Found {len(patients)} patients{bcolors.ENDC}')
    print()

    print(f'{bcolors.OKGREEN}{get_time()} - Processing patients{bcolors.ENDC}')
    print(f'{bcolors.OKGREEN}{"=" * 50}{bcolors.ENDC}')

    df_all = pd.DataFrame()

    start_year = 2000
    for i, patient in enumerate(patients):
        print(f'{bcolors.OKGREEN}{get_time()} - Processing patient {patient}{bcolors.ENDC}')
        print(f'{bcolors.OKCYAN}{"-" * 50}{bcolors.ENDC}')

        patient_data = extract_patient_data(df, patient, datetime(start_year + 2 * i, 1, 1))
        if patient_data is None:
            print(f'{bcolors.FAIL}{get_time()} - Error: Patient {patient} not found{bcolors.ENDC}')
            continue

        print(f'{bcolors.OKCYAN}{get_time()} - Start date: {patient_data.index.min()}{bcolors.ENDC}')
        print(f'{bcolors.OKCYAN}{get_time()} - End date: {patient_data.index.max()}{bcolors.ENDC}')
        print(f'{bcolors.OKCYAN}{get_time()} - Number of rows: {len(patient_data)}{bcolors.ENDC}')
        print()
        print(f'{bcolors.OKCYAN}----- Details {patient} -----------------------------------------------{bcolors.ENDC}')
        print(f'{bcolors.OKCYAN}{patient_data.describe()}{bcolors.ENDC}')
        print(f'{bcolors.OKCYAN}-----------------------------------------------------------------{bcolors.ENDC}')
        print()

        df_all = pd.concat([df_all, patient_data])

    filename = os.path.join(CLEAN_DATA_DIR, 'augmented_data_time_series.csv')
    print(f'{bcolors.OKGREEN}{get_time()} - Save data for all patients into {filename}{bcolors.ENDC}')
    df_all.to_csv(filename)

    print(f'{bcolors.OKGREEN}{get_time()} - Create lag feature data{bcolors.ENDC}')
    print(f'{bcolors.OKGREEN}{"=" * 50}{bcolors.ENDC}')

    # create lagged columns
    # Convert the dictionary to a DataFrame and concatenate with the original
    df_augmented_with_all_lag_features = df_all.copy()

    print(f'{bcolors.OKGREEN}{get_time()} - Create lag feature \'bg+1:00\'{bcolors.ENDC}')
    df_augmented_with_all_lag_features['bg+1:00'] = df_all['bg'].shift(periods=-1, freq="1h")
    for parameter in parameters:
        df_augmented_with_all_lag_features = df_augmented_with_all_lag_features.copy()
        for time_diff in time_diffs:
            print(f'{bcolors.OKGREEN}{get_time()} - Create lag feature \'{parameter}{time_diff}\'{bcolors.ENDC}')
            col_name = f"{parameter}{time_diff}"
            df_augmented_with_all_lag_features[col_name] = df_all[parameter].shift(periods=-1,
                                                                                   freq=parse_time_diff(time_diff))

    # drop all parameter columns
    df_augmented_with_all_lag_features = df_augmented_with_all_lag_features.drop(columns=parameters)

    # set the column order from the train file
    columns = pd.read_csv(os.path.join(RAW_DATA_DIR, 'train.csv'), low_memory=False, index_col=0).columns
    df_augmented_with_all_lag_features = df_augmented_with_all_lag_features[columns]

    filename = os.path.join(CLEAN_DATA_DIR, 'augmented_data_time_series_full_with_lag_features.csv')
    print(f'{bcolors.OKGREEN}{get_time()} - Save data for all patients into {filename}{bcolors.ENDC}')
    df_augmented_with_all_lag_features.to_csv(filename)
    print(f'{bcolors.OKGREEN}{get_time()} - finished - {bcolors.ENDC}')

    print(f'{bcolors.OKGREEN}{get_time()} - Create id column{bcolors.ENDC}')
    # create a id column with: p_num + autoincrement per patient
    df_augmented_with_all_lag_features['id'] = (df_augmented_with_all_lag_features['p_num'] + '_test_' + df_augmented_with_all_lag_features.groupby('p_num').cumcount().astype(str))
    df_augmented_with_all_lag_features = df_augmented_with_all_lag_features.reset_index(drop=True).set_index('id')
    print(f'{bcolors.OKGREEN}{get_time()} - finished - {bcolors.ENDC}')

    # drop all rows where bg-0:00 or bg+1:00 is NaN
    print(f'{bcolors.OKGREEN}{get_time()} - Drop rows with NaN values{bcolors.ENDC}')
    result_df = df_augmented_with_all_lag_features.dropna(subset=['bg+1:00', 'bg-0:00'])
    print(f'{bcolors.OKGREEN}{get_time()} - finished - {bcolors.ENDC}')

    filename = os.path.join(CLEAN_DATA_DIR, 'augmented_4_55h.csv')
    print(f'{bcolors.OKGREEN}{get_time()} - Save data for all patients into {filename}{bcolors.ENDC}')
    result_df_4_55h = result_df.dropna(subset=['bg-4:55'])
    result_df_4_55h.to_csv(filename)
    print(f'{bcolors.OKGREEN}{get_time()} - finished storing {len(result_df_4_55h)} rows - {bcolors.ENDC}')

    filename = os.path.join(CLEAN_DATA_DIR, 'augmented_4_30h.csv')
    print(f'{bcolors.OKGREEN}{get_time()} - Save data for all patients into {filename}{bcolors.ENDC}')
    result_df_4_30h = result_df.dropna(subset=['bg-4:30'])
    result_df_4_30h.to_csv(filename)
    print(f'{bcolors.OKGREEN}{get_time()} - finished storing {len(result_df_4_30h)} rows - {bcolors.ENDC}')

    filename = os.path.join(CLEAN_DATA_DIR, 'augmented_4_00h.csv')
    print(f'{bcolors.OKGREEN}{get_time()} - Save data for all patients into {filename}{bcolors.ENDC}')
    result_df_4h = result_df.dropna(subset=['bg-4:00'])
    result_df_4h.to_csv(filename)
    print(f'{bcolors.OKGREEN}{get_time()} - finished storing {len(result_df_4h)} rows - {bcolors.ENDC}')

    filename = os.path.join(CLEAN_DATA_DIR, 'augmented_3_00h.csv')
    print(f'{bcolors.OKGREEN}{get_time()} - Save data for all patients into {filename}{bcolors.ENDC}')
    result_df_3h = result_df.dropna(subset=['bg-3:00'])
    result_df_3h.to_csv(filename)
    print(f'{bcolors.OKGREEN}{get_time()} - finished storing {len(result_df_3h)} rows - {bcolors.ENDC}')

    filename = os.path.join(CLEAN_DATA_DIR, 'augmented_2_00h.csv')
    print(f'{bcolors.OKGREEN}{get_time()} - Save data for all patients into {filename}{bcolors.ENDC}')
    result_df_2h = result_df.dropna(subset=['bg-2:00'])
    result_df_2h.to_csv(filename)
    print(f'{bcolors.OKGREEN}{get_time()} - finished storing {len(result_df_2h)} rows - {bcolors.ENDC}')

    filename = os.path.join(CLEAN_DATA_DIR, 'augmented_1_00h.csv')
    print(f'{bcolors.OKGREEN}{get_time()} - Save data for all patients into {filename}{bcolors.ENDC}')
    result_df_1h = result_df.dropna(subset=['bg-1:00'])
    result_df_1h.to_csv(filename)
    print(f'{bcolors.OKGREEN}{get_time()} - finished storing {len(result_df_1h)} rows - {bcolors.ENDC}')

    # copy train.csv, test.csv to processed directory
    print(f'{bcolors.OKGREEN}{get_time()} - Copy train.csv to processed directory{bcolors.ENDC}')
    os.system(f'cp {os.path.join(RAW_DATA_DIR, "train.csv")} {CLEAN_DATA_DIR}')
    os.system(f'cp {os.path.join(RAW_DATA_DIR, "test.csv")} {CLEAN_DATA_DIR}')
