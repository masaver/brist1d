from datetime import datetime, timedelta

import pandas as pd

parameters = ['bg', 'insulin', 'carbs', 'hr', 'steps', 'cals', 'activity']
time_diffs = [f'-{i}:{j:02}' for i in range(6) for j in range(0, 60, 5)]


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


def extract_patient_data(df: pd.DataFrame, patient_num: str, start_date: datetime = datetime(2020, 1, 1)) -> pd.DataFrame:
    unique_patient_nums = df['p_num'].unique()
    if patient_num not in unique_patient_nums:
        return None

    df_patient = df[df['p_num'] == patient_num]

    # convert the time column to datetime
    df_patient.loc[:, "time"] = pd.to_datetime(df_patient["time"], format="%H:%M:%S").dt.time

    current_date = start_date
    assigned_dates = []

    for i, row in df_patient.iterrows():
        # add two days it time is before 6:00
        if row['time'] < datetime.strptime("06:00:00", "%H:%M:%S").time():
            current_date = current_date + timedelta(days=1)

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
