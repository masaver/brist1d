
from datetime import datetime, timedelta
from os.path import dirname
import numpy as np
import os
import pandas as pd

def score_model( model , name_tag , top_feats = None , prep_submission = True):
    '''
    This function calculates the train & test scores for a model
    '''

    # Generate predictions
    if top_feats is not None:
        y_preds_train = model.predict( x_train[ top_feats ] )
        y_preds_test = model.predict( x_test[ top_feats ] )
    else:
        y_preds_train = model.predict( x_train )
        y_preds_test = model.predict( x_test )

    # Calculate & print performance scores 
    train_scores = {
        'Set' : 'Train',
        'R2' : r2_score( y_train , y_preds_train ) ,
        'RMSE' : mean_squared_error( y_train , y_preds_train , squared = False )
    }

    test_scores = {
        'Set' : 'Test',
        'R2' : r2_score( y_test , y_preds_test ) ,
        'RMSE' : mean_squared_error( y_test , y_preds_test , squared = False ) 
    }

    print( f'Model: {name_tag}')
    print( train_scores )
    print( test_scores )
    print(' ')
    
    #generate predictions on the 'test.csv' data
    if prep_submission:
        preds = model.predict( df_test )
        preds_df = pd.DataFrame({ 'id' : df_test.index  , 'bg+1:00' : preds })

        # Save the submissions DataFrame
        os.makedirs( './submissions' , exist_ok = True )
        preds_df.to_csv( f'./submissions/{name_tag}.csv' , index = False )
        
def prep_data( df , scaler_obj = None ):
    '''
    The intended use of this function is to prepare the raw/test data from the project
    in order to use it for model training or making predictions
    '''

    # Re-encode the time stamp & Transform it via pd.get_dummies()
    day_phase_transformer = DayPhaseTransformer(  time_column = 'time' , time_format = '%H:%M:%S' , result_column = 'day_phase' )
    df = day_phase_transformer.fit_transform( X = df )
    df_dummies =  pd.get_dummies( df['day_phase'] , dtype=int ) 
    df = pd.concat([ df , df_dummies ] , axis = 1 )
    df = df.drop( 'day_phase' , axis = 1 )

    # Drop un-needed columns
    drop_patterns = ['activity', 'carbs']

    for pattern in drop_patterns:
        drop_columns_transformer = DropColumnsTransformer( starts_with = pattern )
        df = drop_columns_transformer.fit_transform( X = df )

    # Fill NAs
    parameter_strategy = {
        'bg': ['interpolate', 'median'],
        'insulin': ['zero'],
        'cals': ['interpolate', 'median'],
        'hr': ['interpolate', 'median'],
        'steps': ['zero'],
    }

    for parameter, strategy in parameter_strategy.items():
        fill_property_nans_transformer = FillPropertyNaNsTransformer( parameter , strategy ) 
        df = fill_property_nans_transformer.fit_transform( X = df )

    # Fix outliers in the insulin col . Only two values which are megative
    def filter_function(x): return x < 0
    outliers_transformer = PropertyOutlierTransformer('insulin', fill_strategy='zero', filter_function=filter_function)
    df = outliers_transformer.fit_transform( X = df )

    # Drop p_num & time
    df = df.drop( ['p_num','time'] , axis = 1 )

    # Split the date into features and target

    return df

def compare_data( id , metric , train_df ):
    
    '''
    This function is used to compare the raw and parsed data.
    The idea, is for each patient {id} & metric, to compare the set of unique non-NA values
    in both the raw and parsed data
    
    Given that these two sets should be the same, there should be no differences in metrics like mean, std, min & max
    '''
    
    # Filter the raw DataFrame
    df_raw = train_df[ train_df['p_num'] == id ]
    
    # read the parsed data for the given patient ID
    df_parsed = pd.read_csv( f'../data/interim/{id}_train.csv' )
    
    # generate data frames of unique non-NA values
    val1_df = pd.DataFrame({
        'group':'raw',
        'vals':df_raw[df_raw.columns[df_raw.columns.str.startswith(f'{metric}-')]].melt().value.unique()
    })
    
    val2_df = pd.DataFrame({
        'group':'processed',
        'vals':df_parsed[metric].unique()
    })

    val1_df.sort_values( 'vals', inplace = True )
    val2_df.sort_values( 'vals', inplace = True )
    
    # Calculate differences
    vals_df = pd.concat([val1_df,val2_df],axis = 0 )
    vals_df = vals_df.groupby('group').agg(['mean','std',np.min,np.max])
    vals_df = pd.DataFrame( vals_df )
    checks_df  = pd.DataFrame( vals_df.loc['raw',:]-vals_df.loc['processed',:] )
    checks_df.rename( columns = {0:'diff'} , inplace  =True )
    checks_df['p_id'] = id
    checks_df['diff'] = abs( checks_df['diff'] )
    checks_df.reset_index(inplace=True)
    checks_df.drop('level_0',axis=1,inplace=True)
    checks_df.rename( columns = {'level_1':'metric'} , inplace = True )
    checks_df['feature'] =  metric
    checks_df = checks_df[['p_id','feature','metric','diff']]
    checks_df['diff_check'] = checks_df['diff'] < 1e-6

    # Return final DataFrames
    return checks_df , val1_df , val2_df , df_raw , df_parsed


def extract_patient_data_ms(df: pd.DataFrame, patient_num: str, start_date: datetime) -> pd.DataFrame | None:

    '''
    This is a slightly modified verison fo the extract_patient_data function.
    IT includes specific checks to help find out where a problematic value is located
    '''
    
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

    print(f'Check 1: {(df_patient==15.36).any().any()}')

    dfs1 = df_patient.copy()

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
    print(f'Check 1b: {(df_patient==15.36).any().any()}')

    df_patient_combined_values = df_patient[['p_num'] + [f"{parameter}{time_diffs[0]}" for parameter in parameters] + ['bg+1:00']].copy()
    print(f'Check 2 debug: {(df_patient_combined_values==15.36).any().any()}')

    dfs2 = df_patient_combined_values.copy()
    
    df_patient_combined_values = df_patient_combined_values.rename(columns={f"{parameter}{time_diffs[0]}": f"{parameter}" for parameter in parameters})
    df_patient_combined_values = df_patient_combined_values.reindex(
        pd.date_range(start=df_patient.index.min() + parse_time_diff(time_diffs[-1]), end=df_patient.index.max(), freq='5min')
    )

    print(f'Check 2: {(df_patient_combined_values==15.36).any().any()}')

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

    print(f'Check 3: {(df_patient_combined_values==15.36).any().any()}')

    # order the columns
    column_order = ['p_num'] + parameters + ['bg+1:00']
    df_patient_combined_values = df_patient_combined_values[column_order]

    # set patient number and initial resolution
    df_patient_combined_values['p_num'] = patient_num
    df_patient_combined_values['initial_resolution'] = initial_resolution

    print(f'Check 4: {(df_patient_combined_values==15.36).any().any()}')

    return df_patient_combined_values , dfs1 , dfs2

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