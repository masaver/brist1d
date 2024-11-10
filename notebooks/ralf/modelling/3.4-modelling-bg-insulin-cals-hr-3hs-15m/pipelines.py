from sklearn.pipeline import Pipeline

from src.features.transformers import DayPhaseTransformer, DropColumnsTransformer, ExtractColumnsTransformer, FillPropertyNaNsTransformer, GetDummiesTransformer, \
    StandardScalerTransformer, PropertyOutlierTransformer

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
    'insulin-0:00',
    'insulin-0:15',
    'insulin-0:30',
    'insulin-0:45',
    'insulin-1:00',
    'insulin-1:15',
    'insulin-1:30',
    'insulin-1:45',
    'insulin-2:00',
    'insulin-2:15',
    'insulin-2:30',
    'insulin-2:45',
    'insulin-3:00',
    'cals-0:00',
    'cals-0:15',
    'cals-0:30',
    'cals-0:45',
    'cals-1:00',
    'cals-1:15',
    'cals-1:30',
    'cals-1:45',
    'cals-2:00',
    'cals-2:15',
    'cals-2:30',
    'cals-2:45',
    'cals-3:00',
    'hr-0:00',
    'hr-0:15',
    'hr-0:30',
    'hr-0:45',
    'hr-1:00',
    'hr-1:15',
    'hr-1:30',
    'hr-1:45',
    'hr-2:00',
    'hr-2:15',
    'hr-2:30',
    'hr-2:45',
    'hr-3:00',
    'bg+1:00',
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
    ('standard_scaler', StandardScalerTransformer(columns=columns_to_extract[1:-1]))
])

pipeline = Pipeline(steps=[
    ('preprocessing', preprocessing_pipeline),
    ('standardization', standardization_pipeline)
])
