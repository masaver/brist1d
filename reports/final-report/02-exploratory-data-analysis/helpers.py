from datetime import timedelta, datetime

import pandas as pd


def set_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    # parameters
    start_date = datetime(2000, 1, 1)
    ids_with_date_change = []  # p11_4307 is the datapoint that changes the date

    # convert the time column to a datetime object
    df["time"] = pd.to_datetime(df["time"], format="%H:%M:%S").dt.time

    # define temporary variables
    date_times = []
    last_processed_time = None
    last_processed_p_num = None
    current_date = None

    for i, row in df.iterrows():
        # reset when we see a new patient
        if last_processed_p_num is not None and last_processed_p_num != row["p_num"]:
            last_processed_p_num = None
            last_processed_time = None
            current_date = None

        if last_processed_p_num is None:
            last_processed_p_num = row["p_num"]
        if last_processed_time is None:
            last_processed_time = row["time"]
        if current_date is None:
            current_date = start_date

        if row["time"] < last_processed_time or row["id"] in ids_with_date_change:
            current_date = current_date + timedelta(days=1)

        date_times.append(datetime.combine(current_date, row["time"]))
        last_processed_time = row["time"]

    df = df.copy()
    df.loc[:, "datetime"] = date_times
    df = df.set_index("datetime")
    # df = df.drop(columns=["time"])

    return df


def parse_time_diff(time_diff_str: str) -> timedelta:
    """
    :param time_diff_str: string in the format '-HH:MM' or '+HH:MM'
    :return: timedelta
    """
    is_negative = time_diff_str[0] == '-'

    if time_diff_str[0] in ['+', '-']:
        time_diff_str = time_diff_str[1:]

    hours, minutes = time_diff_str.split(':')
    if is_negative:
        hours = -int(hours)
        minutes = -int(minutes)

    return timedelta(hours=int(hours), minutes=int(minutes))


def get_parameters():
    return ['bg', 'insulin', 'carbs', 'hr', 'steps', 'cals', 'activity']


def get_time_diffs():
    return [f'-{i}:{j:02}' for i in range(6) for j in range(0, 60, 5)]


def consistency_check(df: pd.DataFrame) -> dict:
    p_nums = df['p_num'].unique()
    parameters = get_parameters()
    time_diffs = get_time_diffs()

    result_dict = {}
    for p_num in p_nums:
        result_dict[p_num] = {}
        for parameter in parameters:
            df_patient = df[df['p_num'] == p_num]

            new_df = None

            # This is a special case for the target parameter bg+1:00
            if parameter == 'target':
                time_diff_str = '+1:00'
                new_df = df_patient[['id', 'p_num', f'bg-0:00']].copy()
                time_diff = parse_time_diff(time_diff_str)
                values = df_patient['bg+1:00'].copy()
                values.index = values.index + time_diff
                values.index.name = "datetime"

                # add rows that are not in df
                new_df = new_df.reindex(df_patient.index.union(values.index))
                # join the values to the new_df
                new_df = new_df.join(values.rename('bg+1:00-1:00'), on="datetime")

            # This is the general case
            if parameter != 'target':
                new_df = df_patient[['id', 'p_num', f'{parameter}-0:00']].copy()
                for time_diff_id, time_diff_str in enumerate(time_diffs):
                    if time_diff_str == '-0:00':
                        continue

                    if not df_patient.columns.str.contains(f"{parameter}{time_diff_str}").any():
                        continue

                    time_diff = parse_time_diff(time_diff_str)
                    values = df_patient[f"{parameter}{time_diff_str}"].copy()
                    values.index = values.index + time_diff
                    values.index.name = "datetime"

                    # add rows that are not in df
                    new_df = new_df.reindex(df_patient.index.union(values.index))

                    # join the values to the new_df
                    new_df = new_df.join(values.rename(f"{parameter}{time_diff_str}+{time_diff_str}"), on="datetime")

            # analyse the data
            # check for unique values in the columns except for the id and p_num columns
            new_df = new_df.set_index("id", drop=True)
            unique_values = new_df.drop(columns=["p_num"]).nunique(axis=1)
            unique_values = unique_values[unique_values > 1]

            result_dict[p_num][parameter] = unique_values.shape[0]
            result_dict[p_num]['total'] = new_df.shape[0]

    return result_dict
