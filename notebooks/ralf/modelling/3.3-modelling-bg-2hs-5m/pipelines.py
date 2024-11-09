from sklearn.pipeline import Pipeline

from src.features.transformers import DayPhaseTransformer, DropColumnsTransformer, ExtractColumnsTransformer, FillPropertyNaNsTransformer, GetDummiesTransformer, \
    StandardScalerTransformer

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
    'bg-1:05',
    'bg-1:10',
    'bg-1:15',
    'bg-1:20',
    'bg-1:25',
    'bg-1:30',
    'bg-1:35',
    'bg-1:40',
    'bg-1:45',
    'bg-1:50',
    'bg-1:55',
    'bg-2:00',
    'bg+1:00',
]

preprocessing_pipeline = Pipeline(steps=[
    ('day_phase', DayPhaseTransformer(time_column='time', time_format='%H:%M:%S', result_column='day_phase')),
    ('drop_parameter_cols', DropColumnsTransformer(starts_with=['activity', 'carbs', 'steps', 'insulin', 'cals', 'hr'])),
    ('drop_others', DropColumnsTransformer(columns_to_delete=['p_num', 'time'])),
    ('fill_properties_nan_bg', FillPropertyNaNsTransformer(parameter='bg', how=['interpolate', 'median'])),
    ('extract_features', ExtractColumnsTransformer(columns_to_extract=columns_to_extract)),
])

standardization_pipeline = Pipeline(steps=[
    ('get_dummies', GetDummiesTransformer(columns=['day_phase'])),
    ('standard_scaler', StandardScalerTransformer(columns=columns_to_extract[1:-1]))
])

pipeline = Pipeline(steps=[
    ('preprocessing', preprocessing_pipeline),
    ('standardization', standardization_pipeline)
])
