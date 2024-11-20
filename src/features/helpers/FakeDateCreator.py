import pandas as pd

class FakeDateCreator:
    @staticmethod
    def create_pseudo_datetime(patients_df):
        # Define a fake start date
        fake_start_date = '2020-01-01'

        # Convert 'time' to timedelta (duration since midnight)
        patients_df['time_delta'] = pd.to_timedelta(patients_df['time'])

        # Function to create pseudo datetime
        def transform_group(group):
            # Start from the same fake date
            group['pseudo_datetime'] = pd.to_datetime(fake_start_date) + group['time_delta']
            
            # Calculate the days elapsed when time resets (new day)
            group['days_elapsed'] = group['time_delta'].diff().apply(lambda x: 1 if x < pd.Timedelta(0) else 0).cumsum()
            
            # Add days_elapsed to the pseudo_datetime to simulate continuous time
            group['pseudo_datetime'] = group['pseudo_datetime'] + pd.to_timedelta(group['days_elapsed'], unit='D')
            
            return group

        # Apply the transformation
        transformed_df = patients_df.groupby('p_num', group_keys=False).apply(transform_group).reset_index(drop=True)
        
        return transformed_df


    @staticmethod
    def filter_patient_data(patients, p_num, specific_day, days_num=1):
        """
        Filters patient data for a specific patient and day.

        Args:
            patients (pd.DataFrame): The complete patients DataFrame.
            p_num (str): Patient identifier (e.g., 'p01') or 'all' for all patients.
            specific_day (str): Specific day to filter data (format: 'YYYY-MM-DD').

        Returns:
            pd.DataFrame: Filtered data for the given patient and day.
        """
        # If "all", don't filter by patient ID
        if p_num != "all":
            patients = patients[patients['p_num'] == p_num]

        # Ensure 'pseudo_datetime' is in datetime format
        patients_data = patients.copy()
        patients_data['pseudo_datetime'] = pd.to_datetime(patients_data['pseudo_datetime'], errors='coerce')

        # Define the time range for the specific day
        start_datetime = pd.to_datetime(specific_day)
        end_datetime = start_datetime + pd.Timedelta(days=days_num)

        # Filter data within the specific day's time range
        filtered_data = patients_data[
            (patients_data['pseudo_datetime'] >= start_datetime) & 
            (patients_data['pseudo_datetime'] < end_datetime)
        ]

        return filtered_data
