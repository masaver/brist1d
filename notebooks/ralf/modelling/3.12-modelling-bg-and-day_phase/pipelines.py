from sklearn.pipeline import Pipeline

from src.features.transformers import DayPhaseTransformer, DropColumnsTransformer, ExtractColumnsTransformer, FillPropertyNaNsTransformer, GetDummiesTransformer, \
    PropertyOutlierTransformer, StandardScalerTransformer

columns_to_extract = [
    'day_phase',
    'bg-0:00',
    'bg-0:15',
    'bg-0:30',
    'bg-0:45',
    'bg-1:00',
    'bg-1:15',
    'bg-1:30',
    'bg-1:45',
    'bg-2:00',
    'bg-2:15',
    'bg-2:30',
    'bg-2:45',
    'bg-3:00',
    'bg-3:15',
    'bg-3:30',
    'bg-3:45',
    'bg-4:00',
    'bg-4:15',
    'bg-4:30',
    'bg-4:45',
    'bg-5:00',
    'bg-5:15',
    'bg-5:30',
    'bg-5:45',
    'bg+1:00',
]

preprocessing_pipeline = Pipeline(steps=[
    ('day_phase', DayPhaseTransformer(time_column='time', time_format='%H:%M:%S', result_column='day_phase')),
    ('drop_parameter_cols', DropColumnsTransformer(starts_with=['activity', 'carbs', 'steps', 'insulin', 'cals', 'hr'])),
    ('drop_others', DropColumnsTransformer(columns_to_delete=['p_num', 'time'])),
    ('fill_properties_nan_bg', FillPropertyNaNsTransformer(parameter='bg', how=['interpolate'])),
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
