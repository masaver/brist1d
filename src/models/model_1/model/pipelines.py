import numpy as np
from sklearn.pipeline import Pipeline

from src.features.transformers import DayPhaseTransformer, DropColumnsTransformer, FillPropertyNaNsTransformer, GetDummiesTransformer, \
    StandardScalerTransformer, PropertyOutlierTransformer, RenameColumnsTransformer, ExtractColumnsTransformer

cols_2_extract = ['hr_0_00', 'bg_0_15', 'day_phase_evening', 'bg_0_00', 'insulin_0_00', 'day_phase_night', 'bg_0_10']

def filter_function(x): return x < 0

preprocessing_pipeline = Pipeline(steps=[
    ('day_phase', DayPhaseTransformer(time_column='time', time_format='%H:%M:%S', result_column='day_phase')),
    ('drop_parameter_cols', DropColumnsTransformer(starts_with=['activity', 'carbs', 'steps'])),
    ('drop_others', DropColumnsTransformer(columns_to_delete=['p_num', 'time'])),
    ('fill_properties_nan_bg', FillPropertyNaNsTransformer(parameter='bg', how=['interpolate', 'median'])),
    ('fill_properties_nan_insulin', FillPropertyNaNsTransformer(parameter='insulin', how=['zero'])),
    ('fill_properties_nan_cals', FillPropertyNaNsTransformer(parameter='cals', how=['interpolate', 'median'])),
    ('fill_properties_nan_hr', FillPropertyNaNsTransformer(parameter='hr', how=['interpolate', 'median'])),
    ('drop_outliers', PropertyOutlierTransformer(parameter='insulin', filter_function=filter_function, fill_strategy='zero')),
])

standardization_pipeline = Pipeline(steps=[
    ('get_dummies', GetDummiesTransformer(columns=['day_phase'])),
    ('standard_scaler', StandardScalerTransformer(types=[np.float64]))
])

rename_extract_pipeline = Pipeline(steps=[
    ('rename_cols',RenameColumnsTransformer()),
    ('extract_cols',ExtractColumnsTransformer(columns_to_extract=cols_2_extract))
])

pipeline = Pipeline(steps=[
    ('preprocessing', preprocessing_pipeline),
    ('standardization', standardization_pipeline),
    ('rename_extract',rename_extract_pipeline)
])
