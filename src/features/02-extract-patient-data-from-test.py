from datetime import datetime, timedelta
from os.path import dirname
import numpy as np
import os
import pandas as pd
from transformers import FillPropertyNaNsTransformer


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

    for i, row in df_patient.iterrows():
        # each row gets a new date
        current_date = current_date + timedelta(days=1)
        assigned_dates.append(current_date)

    df_patient = df_patient.copy()
    df_patient.loc[:, "date"] = assigned_dates
    df_patient.loc[:, "datetime"] = df_patient.apply(lambda row: datetime.combine(row['date'], row['time']), axis=1)
    df_patient = df_patient.set_index("datetime")
    df_patient = df_patient.drop(columns=["date"])
    df_patient = df_patient.drop(columns=["id"])

    # change the frequency to 5 minutes
    full_date_range = pd.date_range(start=df_patient.index.min(), end=df_patient.index.max(), freq='5min')
    df_patient = df_patient.reindex(full_date_range)
    df_patient.index.name = "datetime"

    # organize the columns
    meta_columns = ['p_num']
    target_columns = ['bg+1:00'] if 'bg+1:00' in df_patient.columns else []

    df_patient_combined_values = df_patient[meta_columns + [f"{parameter}{time_diffs[0]}" for parameter in parameters] + target_columns].copy()
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

    # set patient number and initial resolution
    df_patient_combined_values['p_num'] = patient_num
    df_patient_combined_values['time'] = df_patient_combined_values.index.time

    # order the columns
    meta_columns = meta_columns + ['time']
    column_order = meta_columns + parameters + target_columns
    df_patient_combined_values = df_patient_combined_values[column_order]

    return df_patient_combined_values


if __name__ == '__main__':
    print(f'{bcolors.OKGREEN}{get_time()} - Extracting patient data{bcolors.ENDC}')
    print(f'{bcolors.OKGREEN}{"=" * 50}{bcolors.ENDC}')
    print(f'{bcolors.OKCYAN}{get_time()} - Loading train.csv{bcolors.ENDC}')

    df = pd.read_csv(os.path.join(src_folder, 'test.csv'), na_values=np.nan, low_memory=False)
    df = FillPropertyNaNsTransformer(parameter='bg', how=['interpolate', 'median'], precision=2).fit_transform(df)

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
        print()

        df_all = pd.concat([df_all, patient_data])

    print(f'{bcolors.OKGREEN}{get_time()} - Save data for all patients into all_test_combined.csv{bcolors.ENDC}')
    df_all.to_csv(os.path.join(interim_folder, 'all_test_combined.csv'))

    print(f'{bcolors.OKGREEN}{get_time()} - Create lag feature data{bcolors.ENDC}')
    print(f'{bcolors.OKGREEN}{"=" * 50}{bcolors.ENDC}')

    # Drop all rows where bg is NaN
    df_all = df_all.dropna(subset=['bg'])

    # create lagged columns
    # Convert the dictionary to a DataFrame and concatenate with the original
    patient_ids = df_all['p_num'].unique()
    result_df = None
    for patient_id in patient_ids:
        new_columns = {}
        patient_data = df_all[df_all['p_num'] == patient_id]
        new_columns['bg+1:00'] = patient_data['bg'].shift(periods=-1, freq="1h")
        for parameter in parameters:
            for time_diff in time_diffs:
                col_name = f"{parameter}{time_diff}"
                new_columns[col_name] = patient_data[parameter].shift(periods=-1, freq=parse_time_diff(time_diff))

        new_data = pd.DataFrame(new_columns)
        patient_data = pd.concat([patient_data, new_data], axis=1)
        if result_df is None:
            result_df = patient_data
        else:
            result_df = pd.concat([result_df, patient_data])

    # drop all parameter columns
    result_df = result_df.drop(columns=parameters)

    # read columns from the original train
    columns = pd.read_csv(os.path.join(src_folder, 'train.csv'), low_memory=False, index_col=0).columns
    result_df = result_df[columns]

    # drop all rows where bg-0:00 or bg+1:00 is NaN
    result_df = result_df.dropna(subset=['bg+1:00', 'bg-0:00'])

    result_df_5h = result_df.dropna(subset=['bg-5:00'])
    result_df_5h.to_csv(os.path.join(interim_folder, 'all_test_5h.csv'), index=False)

    result_df_4h = result_df.dropna(subset=['bg-4:00'])
    result_df_4h.to_csv(os.path.join(interim_folder, 'all_test_4h.csv'), index=False)

    result_df_3h = result_df.dropna(subset=['bg-3:00'])
    result_df_3h.to_csv(os.path.join(interim_folder, 'all_test_3h.csv'), index=False)

    result_df_2h = result_df.dropna(subset=['bg-2:00'])
    result_df_2h.to_csv(os.path.join(interim_folder, 'all_test_2h.csv'), index=False)

    result_df_1h = result_df.dropna(subset=['bg-1:00'])
    result_df_1h.to_csv(os.path.join(interim_folder, 'all_test_1h.csv'), index=False)
