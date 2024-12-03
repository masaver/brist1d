from sklearn.pipeline import Pipeline

from src.features.transformers import DropColumnsTransformer, ExtractColumnsTransformer, FillPropertyNaNsTransformer, GetDummiesTransformer, PropertyOutlierTransformer, \
    StandardScalerTransformer, DateTimeHourTransformer

columns_to_extract = [
    'p_num',
    'hour_sin',
    'hour_cos',
    'bg-0:00',
    'bg-0:05',
    'bg-0:10',
    'bg-0:15',
    'bg-0:20',
    'bg-0:25',
    'bg-0:30',
    'bg-0:35',
    'bg-0:40',
    'bg-0:45',
    'bg-0:50',
    'bg-0:55',
    'bg-1:00',
    'insulin-0:00',
    'insulin-0:05',
    'insulin-0:10',
    'insulin-0:15',
    'insulin-0:20',
    'insulin-0:25',
    'insulin-0:30',
    'insulin-0:35',
    'insulin-0:40',
    'insulin-0:45',
    'insulin-0:50',
    'insulin-0:55',
    'insulin-1:00',
    'cals-0:00',
    'hr-0:00',
    'steps-0:00',
    'bg+1:00'
]

preprocessing_pipeline = Pipeline(steps=[
    ('date_time', DateTimeHourTransformer(time_column='time', result_column='hour', type='sin_cos', drop_time_column=True)),
    ('drop_parameter_cols', DropColumnsTransformer(starts_with=['activity', 'carbs'])),
    ('drop_others', DropColumnsTransformer(columns_to_delete=['time'])),
    ('fill_properties_nan_bg', FillPropertyNaNsTransformer(parameter='bg', how=['interpolate', 'median'], interpolate=3, ffill=1, bfill=1, precision=1)),
    ('fill_properties_nan_insulin', FillPropertyNaNsTransformer(parameter='insulin', how=['interpolate', 'median'], interpolate=3, ffill=1, bfill=1, precision=4)),
    ('fill_properties_nan_cals', FillPropertyNaNsTransformer(parameter='cals', how=['interpolate', 'median'], interpolate=3, ffill=1, bfill=1, precision=1)),
    ('fill_properties_nan_hr', FillPropertyNaNsTransformer(parameter='hr', how=['interpolate', 'median'], interpolate=3, ffill=1, bfill=1, precision=1)),
    ('fill_properties_nan_steps', FillPropertyNaNsTransformer(parameter='steps', how=['zero'], interpolate=3, ffill=1, bfill=1, precision=1)),
    ('drop_outliers', PropertyOutlierTransformer(parameter='insulin', filter_function=lambda x: x < 0, fill_strategy='zero')),
    ('extract_features', ExtractColumnsTransformer(columns_to_extract=columns_to_extract)),
])

standardization_pipeline = Pipeline(steps=[
    ('get_dummies', GetDummiesTransformer(columns=['hour', 'p_num'])),
    ('standard_scaler', StandardScalerTransformer(columns=columns_to_extract[1:-1]))
])

pipeline = Pipeline(steps=[
    ('preprocessing', preprocessing_pipeline),
    ('standardization', standardization_pipeline)
])
