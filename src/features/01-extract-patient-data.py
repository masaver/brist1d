from datetime import datetime, timedelta
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


def parse_time_diff(time_diff_str: str) -> timedelta:
    # get a string in the format of -HH:MM and return a timedelta object
    is_negative = time_diff_str[0] == '-'

    if time_diff_str[0] in ['+', '-']:
        time_diff_str = time_diff_str[1:]

    hours, minutes = time_diff_str.split(':')
    if is_negative:
        hours = -int(hours)
        minutes = -int(minutes)

    return timedelta(hours=int(hours), minutes=int(minutes))


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

    # get resolution of the data
    initial_resolution_in_seconds = (df_patient.index[1] - df_patient.index[0]).seconds
    initial_resolution_in_minutes = initial_resolution_in_seconds / 60
    initial_resolution = f"{int(initial_resolution_in_minutes)}min"

    # change the frequency to 5 minutes
    full_date_range = pd.date_range(start=df_patient.index.min(), end=df_patient.index.max(), freq='5min')
    df_patient = df_patient.reindex(full_date_range)
    df_patient.index.name = "datetime"

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

    df_patient_combined_values = df_patient[['p_num'] + [f"{parameter}{time_diffs[0]}" for parameter in parameters] + ['bg+1:00']].copy()
    df_patient_combined_values = df_patient_combined_values.rename(columns={f"{parameter}{time_diffs[0]}": f"{parameter}" for parameter in parameters})
    df_patient_combined_values = df_patient_combined_values.reindex(
        pd.date_range(start=df_patient.index.min() + parse_time_diff(time_diffs[-1]), end=df_patient.index.max(), freq='5min')
    )

    for parameter in parameters:
        for time_diff_id, time_diff_str in enumerate(time_diffs):
            if time_diff_str == '-0:00':
                continue

            if not df_patient.columns.str.contains(f"{parameter}{time_diff_str}").any():
                continue

            time_diff = parse_time_diff(time_diff_str)
            values = df_patient[f"{parameter}{time_diff_str}"].copy()
            values.index = values.index + time_diff
            df_patient_combined_values[parameter] = df_patient_combined_values[parameter].combine_first(values)

    print(df_patient_combined_values.columns)

    # order the columns
    column_order = ['p_num'] + parameters + ['bg+1:00']
    df_patient_combined_values = df_patient_combined_values[column_order]

    # set patient number and initial resolution
    df_patient_combined_values['p_num'] = patient_num
    df_patient_combined_values['initial_resolution'] = initial_resolution

    return df_patient_combined_values


if __name__ == '__main__':
    print(f'{bcolors.OKGREEN}{get_time()} - Cleanup interim folder{bcolors.ENDC}')
    print(f'{bcolors.OKGREEN}{"=" * 50}{bcolors.ENDC}')
    for file in os.listdir(interim_folder):
        if file.endswith(".csv"):
            os.remove(os.path.join(interim_folder, file))
            print(f'{bcolors.OKCYAN}{get_time()} - Delete {file}{bcolors.ENDC}')

    print()
    print(f'{bcolors.OKGREEN}{get_time()} - Extracting patient data{bcolors.ENDC}')
    print(f'{bcolors.OKGREEN}{"=" * 50}{bcolors.ENDC}')
    print(f'{bcolors.OKCYAN}{get_time()} - Loading train.csv{bcolors.ENDC}')
    df = pd.read_csv(os.path.join(src_folder, 'train.csv'), na_values=np.nan, low_memory=False)
    patients = df['p_num'].unique()
    print(f'{bcolors.OKCYAN}{get_time()} - Found {len(patients)} patients{bcolors.ENDC}')
    print()

    print(f'{bcolors.OKGREEN}{get_time()} - Processing patients{bcolors.ENDC}')
    print(f'{bcolors.OKGREEN}{"=" * 50}{bcolors.ENDC}')

    df_all = pd.DataFrame()

    for patient in patients:
        print(f'{bcolors.OKGREEN}{get_time()} - Processing patient {patient}{bcolors.ENDC}')
        print(f'{bcolors.OKCYAN}{"-" * 50}{bcolors.ENDC}')

        patient_data = extract_patient_data(df, patient, datetime(2020, 1, 1))
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

        patient_data.to_csv(os.path.join(interim_folder, f'{patient}_train.csv'))
        print(f'{bcolors.OKCYAN}{get_time()} - {patient}_train.csv stored to interim folder{bcolors.ENDC}')
        print()

        df_all = pd.concat([df_all, patient_data])

    print(f'{bcolors.OKGREEN}{get_time()} - Save data for all patients into all_train.csv{bcolors.ENDC}')
    df_all.to_csv(os.path.join(interim_folder, 'all_train.csv'))
