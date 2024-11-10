from sklearn.pipeline import Pipeline

from src.features.transformers import DayPhaseTransformer, DropColumnsTransformer, ExtractColumnsTransformer, FillPropertyNaNsTransformer, GetDummiesTransformer, \
    PropertyOutlierTransformer, StandardScalerTransformer

columns_to_extract = [
    'day_phase',
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
    'cals-0:00',
    'cals-0:05',
    'cals-0:10',
    'cals-0:15',
    'cals-0:20',
    'cals-0:25',
    'cals-0:30',
    'hr-0:00',
    'hr-0:05',
    'hr-0:10',
    'hr-0:15',
    'hr-0:20',
    'hr-0:25',
    'hr-0:30',
    'bg+1:00'
]

preprocessing_pipeline = Pipeline(steps=[
    ('day_phase', DayPhaseTransformer(time_column='time', time_format='%H:%M:%S', result_column='day_phase')),
    ('drop_parameter_cols', DropColumnsTransformer(starts_with=['activity', 'carbs', 'steps'])),
    ('drop_others', DropColumnsTransformer(columns_to_delete=['p_num', 'time'])),
    ('fill_properties_nan_bg', FillPropertyNaNsTransformer(parameter='bg', how=['interpolate', 'median'])),
    ('fill_properties_nan_insulin', FillPropertyNaNsTransformer(parameter='insulin', how=['zero'])),
    ('fill_properties_nan_cals', FillPropertyNaNsTransformer(parameter='cals', how=['interpolate', 'median'])),
    ('fill_properties_nan_hr', FillPropertyNaNsTransformer(parameter='hr', how=['interpolate', 'median'])),
    ('drop_outliers', PropertyOutlierTransformer(parameter='insulin', filter_function=lambda x: x < 0, fill_strategy='zero')),
    ('extract_features', ExtractColumnsTransformer(columns_to_extract=columns_to_extract)),
])

standardization_pipeline = Pipeline(steps=[
    ('get_dummies', GetDummiesTransformer(columns=['day_phase'])),
    ('min_max_scaler', StandardScalerTransformer(columns=columns_to_extract[1:-1]))
])

pipeline = Pipeline(steps=[
    ('preprocessing', preprocessing_pipeline),
    ('standardization', standardization_pipeline)
])
