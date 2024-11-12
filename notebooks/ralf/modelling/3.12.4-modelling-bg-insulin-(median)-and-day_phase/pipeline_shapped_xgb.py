from sklearn.pipeline import Pipeline

from src.features.transformers import DayPhaseTransformer, DropColumnsTransformer, ExtractColumnsTransformer, FillPropertyNaNsTransformer, GetDummiesTransformer, \
    PropertyOutlierTransformer, StandardScalerTransformer

shapped_columns_XG_BOOST = [
    'bg-0:00',
    'bg-0:05',
    'bg-0:15',
    'bg-0:10',
    'insulin-0:05',
    'insulin-0:00',
    'bg-5:55',
    'bg-0:20',
    'insulin-0:30',
    'insulin-5:55',
    'insulin-0:10',
    'bg-0:30',
    'insulin-0:20',
    'insulin-0:45',
    'insulin-0:15',
    'insulin-4:30',
    'insulin-0:50',
    'bg-0:45',
    'insulin-0:25',
    'insulin-1:05',
    'bg-0:25',
    'insulin-0:55',
    'bg-1:45',
    'insulin-5:30',
    'insulin-5:50',
    'insulin-5:25',
    'insulin-1:45',
    'bg-3:30',
    'insulin-1:30',
    'bg-3:20',
    'insulin-1:15',
    'bg-3:55',
    'bg-2:15',
    'insulin-4:55',
    'bg-4:45',
    'insulin-5:45',
    'day_phase_noon',
    'insulin-4:50',
    'bg-2:05',
    'bg-3:10',
    'bg-1:30',
    'insulin-2:30',
    'insulin-5:15',
    'bg-2:20',
    'bg-1:20',
    'bg-5:50',
    'bg-2:50',
    'bg-2:30',
    'insulin-4:45',
    'insulin-1:00',
    'insulin-1:10',
    'bg-1:25',
    'bg-5:10',
    'bg-4:20',
    'insulin-4:05',
    'bg-1:00',
    'bg-5:30',
    'insulin-5:00',
    'bg-5:00',
    'insulin-2:45',
    'insulin-2:00',
    'insulin-4:15',
    'bg-3:45',
    'bg-4:00',
    'insulin-3:30',
    'bg-1:15',
    'insulin-3:55']

columns_to_extract = ['day_phase'] + shapped_columns_XG_BOOST + ['bg+1:00']

preprocessing_pipeline = Pipeline(steps=[
    ('day_phase', DayPhaseTransformer(time_column='time', time_format='%H:%M:%S', result_column='day_phase')),
    ('drop_parameter_cols', DropColumnsTransformer(starts_with=['activity', 'carbs', 'steps', 'cals', 'hr'])),
    ('drop_others', DropColumnsTransformer(columns_to_delete=['p_num', 'time'])),
    ('fill_properties_nan_bg', FillPropertyNaNsTransformer(parameter='bg', how=['interpolate'])),
    ('fill_properties_nan_insulin', FillPropertyNaNsTransformer(parameter='insulin', how=['median'])),
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
