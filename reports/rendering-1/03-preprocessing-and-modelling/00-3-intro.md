# Preprocessing and Modelling

The objective of this section is to explain the preprocessing steps and the preliminary modelling that was done in the context of the Kaggle competition.
The preprocessing steps are crucial to ensure that the data is in a format that can be used for modelling.
The preliminary modelling is done to understand the data better and to get a sense of the performance of the models.

## Preprocessing steps

In this section we outline the pre-processing steps that we think will be required in order to correctly format the data before different forms of modelling.

We wrote a bunch of [transformers](src/features/transformers) highly adapted to our data. We will use them to preprocess the data before modelling and before predicting.

see: [Preprocessing steps](01-preprocessing-steps.md)

## Preliminary Modelling

In this section we outline the preliminary modelling that we have done to understand the data better and to get a sense of the performance of the models.

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
