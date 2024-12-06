# Exploratory Data Analysis Conclusions

* Anomalies:
    * A minor anomaly was detected for ``p12``, where two negativ values were identified as irregular.
    * Only one patient (``p11``) shows small inconsistencies in continuous rows within the raw data.
* Missing Data:
    * Feature columns in the raw data generally have moderate fractions of missing values (NAs). 
    * The columns related to ``activity`` and ``carbs`` have a substantial missing values with over 95%.
* Feature Distributions:
    * With only minor exceptions, the distribution of features and target variable is largely comparable across patients.
    * All variables exhibit right-skewed distributions.
* Correlations:
    * Most features, whether considered as whole time series or lagged features, show weak correlations with ``bg+1:00``, emphasizing the predictive dominance of recent ``bg`` values.


## Business Relevant Insights

* **Data challenges:**
The dataset provides valuable variables known to influence blood glucose levels, but using some of them directly is challenging. For instance, columns related to activity and carbohydrates have over 95% missing values, making them difficult to include in modelling without substantial data imputation. Additionally, the activity data includes multiple activity types, requiring specialized encoding methods to make the data usable.

* **Predictive power of prior blood glucose:**
Prior blood glucose (``bg``) levels are strongly correlated with future glucose (``bg+1:00``), while other variables show weak correlations. This suggests that a well-performing model could be built using only prior glucose levels, potentially reducing the need for additional data or complex devices. However, integrating other variables could still enhance the model's performance if the challenges related to their data quality and usability are resolved.

This analysis underscores the importance of balancing data complexity with practicality in designing products or technologies for blood glucose prediction.


## Planned Steps for Upcomming Analysis

### Preprocessing & Preliminary Modelling:
1. Raw data preprocessing
    * Impute the raw data using the column median (to address skewed distributions) and run LazyPredict.
2. Processed data preprocessing
    * Impute the processed data ("long format") using either the column median or time series interpolation and run LazyPredict.


### Note/Reference
(Lazy Predict)[https://github.com/shankarpandala/lazypredict] is a Python library designed to build a variety of basic models for both classification and regression tasks with minimal coding effort. It allows for a quick comparison of models to understand which ones perform better on a given dataset, all without requiring parameter tuning. This makes it a valuable tool for exploratory modelling and baseline performance evaluation.