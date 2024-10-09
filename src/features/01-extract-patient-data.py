from datetime import datetime, timedelta
from ftplib import print_line
from os.path import dirname

import numpy as np
import os
import pandas as pd


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
src_folder = os.path.join(script_directory, '..', '..', 'data', 'raw')
interim_folder = os.path.join(script_directory, '..', '..', 'data', 'interim')


def get_time() -> str:
    return datetime.now().strftime("%H:%M:%S")


def extract_patient_data(df: pd.DataFrame, patient_num: str, start_date: datetime) -> pd.DataFrame | None:
    if patient_num not in df['p_num'].unique():
        return None

    df_patient = df[df['p_num'] == patient_num]

    # convert the time column to datetime
    df_patient.loc[:, "time"] = pd.to_datetime(df_patient["time"], format="%H:%M:%S").dt.time

    current_date = start_date
    assigned_dates = []
    last_time = None

    for i, row in df_patient.iterrows():
        if last_time is None:
            last_time = row["time"]

        if row["time"] < last_time:
            current_date = current_date + timedelta(days=1)

        assigned_dates.append(current_date)
        last_time = row["time"]

    df_patient = df_patient.copy()
    df_patient.loc[:, "date"] = assigned_dates
    df_patient.loc[:, "datetime"] = df_patient.apply(lambda row: datetime.combine(row['date'], row['time']), axis=1)
    df_patient.set_index("datetime", inplace=True)
    df_patient = df_patient.drop(columns=["date", "time", "id"])

    # change the frequency to 5 minutes
    full_date_range = pd.date_range(start=df_patient.index.min(), end=df_patient.index.max(), freq='5min')
    df_patient = df_patient.reindex(full_date_range)
    df_patient.index.name = "datetime"

    # drop the id column and add the patient number
    df_patient = df_patient.copy()
    df_patient.loc[:, "p_num"] = patient_num

    # organize the columns
    parameters = ['bg', 'insulin', 'carbs', 'hr', 'steps', 'cals', 'activity']
    time_diffs = [
        '-0:00',
        '-0:05',
        '-0:10',
        '-0:15',
        '-0:20',
        '-0:25',
        '-0:30',
        '-0:35',
        '-0:40',
        '-0:45',
        '-0:50',
        '-0:55',
        '-1:00',
        '-1:05',
        '-1:10',
        '-1:15',
        '-1:20',
        '-1:25',
        '-1:30',
        '-1:35',
        '-1:40',
        '-1:45',
        '-1:50',
        '-1:55',
        '-2:00',
        '-2:05',
        '-2:10',
        '-2:15',
        '-2:20',
        '-2:25',
        '-2:30',
        '-2:35',
        '-2:40',
        '-2:45',
        '-2:50',
        '-2:55',
        '-3:00',
        '-3:05',
        '-3:10',
        '-3:15',
        '-3:20',
        '-3:25',
        '-3:30',
        '-3:35',
        '-3:40',
        '-3:45',
        '-3:50',
        '-3:55',
        '-4:00',
        '-4:05',
        '-4:10',
        '-4:15',
        '-4:20',
        '-4:25',
        '-4:30',
        '-4:35',
        '-4:40',
        '-4:45',
        '-4:50',
        '-4:55',
        '-5:00',
        '-5:05',
        '-5:10',
        '-5:15',
        '-5:20',
        '-5:25',
        '-5:30',
        '-5:35',
        '-5:40',
        '-5:45',
        '-5:50',
        '-5:55'
    ]

    for parameter in parameters:
        for time_diff_id, t_diff in enumerate(time_diffs):
            if time_diff_id == 0:
                df_patient.loc[:, parameter] = df_patient[f'{parameter}-0:00']
                df_patient = df_patient.drop(columns=[f'{parameter}-0:00'])
                continue
            if not df_patient.columns.str.contains(f"{parameter}{t_diff}").any():
                continue

            # shift the columns by the time difference and fill the last values with nan
            df_patient.loc[:, f"{parameter}{t_diff}"] = df_patient[f"{parameter}{t_diff}"].shift(-time_diff_id)
            df_patient.loc[df_patient.index[-time_diff_id:], f"{parameter}{t_diff}"] = np.nan

            # fill the parameter column with the first non-nan value
            df_patient[parameter] = df_patient[f'{parameter}'].combine_first(df_patient[f'{parameter}{t_diff}'])
            df_patient = df_patient.drop(columns=[f'{parameter}{t_diff}'])

    # order the columns
    new_column_order = ['p_num'] + parameters + ['bg+1:00']
    df_patient = df_patient[new_column_order]

    return df_patient


if __name__ == '__main__':
    print()
    print(f'{bcolors.OKGREEN}{get_time()} - Extracting patient data{bcolors.ENDC}')
    print(f'{bcolors.OKGREEN}{"=" * 50}{bcolors.ENDC}')
    print()
    print(f'{bcolors.OKGREEN}{get_time()} - Reading test-data data{bcolors.ENDC}')
    df = pd.read_csv(os.path.join(src_folder, 'train.csv'), na_values=np.nan, low_memory=False)
    patients = df['p_num'].unique()
    print(f'{bcolors.OKGREEN}{"-" * 50}{bcolors.ENDC}')
    print(f'{bcolors.OKCYAN}{get_time()} - Found {len(patients)} patients{bcolors.ENDC}')
    print()

    print(f'{bcolors.OKGREEN}{get_time()} - Processing patients{bcolors.ENDC}')
    print(f'{bcolors.OKGREEN}{"=" * 50}{bcolors.ENDC}')
    print()

    for patient in patients:
        print(f'{bcolors.OKGREEN}{get_time()} - Processing patient {patient}{bcolors.ENDC}')
        print(f'{bcolors.OKGREEN}{"-" * 50}{bcolors.ENDC}')
        patient_data = extract_patient_data(df, patient, datetime(2020, 1, 1))
        if patient_data is None:
            print(f'{bcolors.FAIL}{get_time()} - Error: Patient {patient} not found{bcolors.ENDC}')
            continue

        print(f'{bcolors.OKCYAN}{get_time()} - Start date: {patient_data.index.min()}{bcolors.ENDC}')
        print(f'{bcolors.OKCYAN}{get_time()} - End date: {patient_data.index.max()}{bcolors.ENDC}')
        print(f'{bcolors.OKCYAN}{get_time()} - Number of rows: {len(patient_data)}{bcolors.ENDC}')

        print(bcolors.OKCYAN)
        print(patient_data.describe())
        print(bcolors.ENDC)

        patient_data.to_csv(os.path.join(interim_folder, f'{patient}.csv'))
        print(f'{bcolors.OKCYAN}{get_time()} - {patient}.csv stored to interim folder{bcolors.ENDC}')
        print()
