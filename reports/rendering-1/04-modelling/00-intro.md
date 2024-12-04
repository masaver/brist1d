# Modelling

In this section we show and describe two different approaches to modelling the data. The first approach is mainly based on the train data given by the competition, while the second
approach uses the test data to create augmented train data.

## Approaches based on the raw data

We found two possible approaches to preprocess the data and model it.

1. **Traditional Train Data Approach**

We used the train data (from 8 patients) to create a global model that can be used to predict the blood glucose levels for all the participants.

2. **Train and Test Data Approach** using the train data

As the traditional approach does not take in consideration the individual characteristics of each patient, we will take into account the test data to get more information about the
individual characteristics of each patient.
As the given data is based on lag features for each parameter, we can create synthetic train data from test data, shifting the lag features to the right and using previous hours of
data to predict the next hours.

## Steps for the two approaches

For both approaches, we will use the following steps:

* Relevant preprocessing steps
* Running LazyPredict with a large subset of the features (up to 300 columns)
* Defining the most important features with SHAP or other feature importance methods
* Get the most promising models running LazyPredict with the most important features
* Hyperparameter tuning with SciKit-Optimize
* Ensemble models
* Predicting the test data
