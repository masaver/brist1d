# Preprocessing steps

In this section we outline the pre-processing steps that we think will be required in order to correctly format the data before different forms of modelling.

## Transformers

We wrote a bunch of [transformers](https://scikit-learn.org/stable/data_transforms.html) highly adapted to our data. We will use them to preprocess the data before modelling and before predicting.

* DataTimeHourTransformer
* DateTimeTransformer
* DayPhaseTransformer
* DropColumnsTransformer
* ExtractColumnsTransformer
* FillPropertyNaNsTransformer
* GetDummiesTransformer
* Log1pTransformer
* MinMaxScalerTransformer
* PropertyOutlierTransformer
* RollingAverageTransformer
* StandardScalerTransformer

In each model we can combine these transformers freely in a `preprocessing pipeline` and a `scaling pipeline`.

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

Based on our exploratory data analysis, the following preprocessing steps are the most appropriate:

* Columns related to **calories** and **activity are initially removed, as they have very large fractions of NAs (>95%)
* For **p12** two values in the insulin columns are negative. We can fix them by replacing them with `zero` or the `median` of the column.

This leaves us with the lag features for the following numerical parameters:

* **bg** - Blood Glucose
* **insulin** - Insulin
* **carbs** - Carbohydrates
* **hr** - Heart Rate
* **steps** - Steps
* **cals** - Calories

For imputing the missing values prior to modelling, we plan to first do a train/test split and then use the medians of the train set to impute the columns of both the train and
test set. This columns should also be standardized ( e.g: with StandardScaler() ).

Once we have established a 'good-performing' model, we can see if adding the information in the columns related to activity & carbs further improves the model. Furthermore, the
information contained in the timestamp column could be used to create a categorical column with the time of the day (e.g: Morning, Afternoon, etc ... )

