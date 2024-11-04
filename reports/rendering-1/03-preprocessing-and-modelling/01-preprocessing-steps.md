# Preprocessing steps

In this section we outline the pre-processing steps that we think will be requiered in order to correctly format the data before different forms of modelling.

Based on our exploratory data analysis, the following preprocessing steps are the most appropiate:
* Columns related to **calories** and **activity are initially removed, as they have very large fractions of NAs (>95%)

* For **p12** two values in the insulin columns are negative. We can fix them by replacing them with the median of this feature

This leaves us with the lag features related to the following metrics: **['bg','insulin','carbs','hr','steps','cals']**, which are all numerical. For imputing the missing values prior to modelling, we plan to first do a train/test split and then use the medians of the train set to impute the columns of both the train and test set. This columns should also be standarized ( e.g: with StandardScaler() ).

Once we have established a 'good-performing' model, we can see if adding the information in the  columns related to activity & carbs further improves the model. Furthermore, the information contained in the timestamp column  could be used to create a categorical column with the time of the day (e.g: Morning, Afternoon, etc ... )