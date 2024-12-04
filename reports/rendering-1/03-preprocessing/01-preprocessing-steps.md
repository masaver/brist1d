# Preprocessing Steps

In this section, we outline the pre-processing steps that in our opinion are required to format the data properly for different modelling approaches.

## Transformers

Following custom [transformers](https://scikit-learn.org/stable/data_transforms.html) were developed that are designed to handle the unique characteristics of our dataset. They can be then used in preprocessing pipelines for both training and prediction phases.

* ``DataTimeHourTransformer``: Encodes time-of-day information as either sine and cosine values or discrete bins.
* ``DateTimeTransformer``: Converts the timestamp column in a DateTime format and optionally renames the column allowing the original column to be dropped.
* ``DayPhaseTransformer``: Categorizes timestamps into day phases (e.g., morning, noon, night) based on predefined ranges.
* ``DropColumnsTransformer``: Removes specified columns or columns whose names start with given prefixes from a DataFrame (e.g. ``carbs-*``).
* ``ExtractColumnsTransformer``: Selects specific columns for modelling.
* ``FillPropertyNaNsTransformer``: Imputes missing values (NaNs) for specific variables (e.g., ``bg``, ``insulin``) using various methods such as mean, median, zero, or interpolation and forward/backward fill options.
* ``GetDummiesTransformer``: Converts categorical features into one-hot encoded format.
* ``Log1pTransformer``: Applies a logarithmic transformation to reduce skewness in numerical features.
* ``MinMaxScalerTransformer``: Scales numerical features to a defined range, typically [0, 1].
* ``PropertyOutlierTransformer``: Detects outliers in columns related to a specified parameter using a user-defined filter function and replaces them using a configurable fill strategy such as zero, min, max, mean, or median.
* ``RollingAverageTransformer``: Computes rolling averages for time-series features to smooth fluctuations.
* ``StandardScalerTransformer``: Standardizes numerical features by removing the mean and scaling to unit variance.

For each model, these transformers can be combined flexibly into a ``preprocessing pipeline`` and a ``scaling pipeline`` to perform the data preparation to the specific needs of the model.

``` python
preprocessing_pipeline = Pipeline(steps=[
    ('date_time', DateTimeTransformer(source_column='time', source_time_format='%H:%M:%S', target_column='hour', target_time_format='%H')),
    ('drop_parameter_cols', DropColumnsTransformer(starts_with=['activity', 'carbs'])),
    ('drop_others', DropColumnsTransformer(columns_to_delete=['p_num', 'time'])),
    ('fill_properties_nan_bg', FillPropertyNaNsTransformer(parameter='bg', how=['interpolate', 'median'])),
    ('fill_properties_nan_insulin', FillPropertyNaNsTransformer(parameter='insulin', how=['zero'])),
    ('fill_properties_nan_cals', FillPropertyNaNsTransformer(parameter='cals', how=['interpolate', 'median'])),
    ('fill_properties_nan_hr', FillPropertyNaNsTransformer(parameter='hr', how=['interpolate', 'median'])),
    ('fill_properties_nan_steps', FillPropertyNaNsTransformer(parameter='steps', how=['zero'])),
    ('drop_outliers', PropertyOutlierTransformer(parameter='insulin', filter_function=lambda x: x < 0, fill_strategy='zero')),
    ('extract_features', ExtractColumnsTransformer(columns_to_extract=columns_to_extract)),
])

standardization_pipeline = Pipeline(steps=[
    ('get_dummies', GetDummiesTransformer(columns=['hour'])),
    ('min_max_scaler', StandardScalerTransformer(columns=columns_to_extract[2:-1]))
])

pipeline = Pipeline(steps=[
    ('preprocessing', preprocessing_pipeline),
    ('standardization', standardization_pipeline)
])
```

## Preprocessing steps

Based on the insights gained from exploratory data analysis, the following preprocessing steps have been identified as the most appropriate for preparing the dataset for modeling:

1. Handling columns with high missing values:
    * Columns related to ``carbs`` and ``activity`` are removed due to their high fraction of missing values (>95%).
2. Fixing anomalies:
    * For Patient ``p12``, two negative values in the insulin column will be corrected by replacing them with zero.
3. Selected features for modelling:
    * The lag features of the following numerical parameters will be retained for modelling:
        * ``bg``: Blood glucose
        * ``insulin``: Insulin
        * ``carbs``: Carbohydrates
        * ``hr``: Heart Rate
        * ``steps``: Steps
        * ``cals``: Calories
4. Imputation and standardization:
    * Missing values will be imputed after splitting the data into train and test sets. We'll use interpolation and medians techniques  for both train and test sets.
    * Numerical columns will be standardized, for example, using StandardScaler(), to ensure consistent scaling across features.
5. Feature Engineering:
    * Additional features, such as a time of day categorical column (e.g., Morning, Afternoon, Evening), can be created from the timestamp data to provide contextual information.
    * Once a well-performing model is established, the dataset can be revisited to incorporate the removed activity and carbohydrate columns to evaluate their impact on model performance.
    
