# Modelling

In this section, we present and explain two different approaches to modelling the data. The first approach is based on the train data given by the competition, while the second one uses the test data for a creation of augmented train data.

## Approaches Based on the Raw Data

We drove two different approaches to preprocess the data and to model them.

1. **Traditional Train Data Approach**
* A global model was created using only the given train data (from 8 patients) to predict blood glucose levels for all participants.
* This approach does not account for individual patient characteristics but provides a baseline model for comparison.

2. **Train and Test Data Approach** 
* To address the limitations of the traditional approach, this method incorporates test data to capture individual patient characteristics.
* Synthetic train data were created from the test data by shifting lag features to the right, using previous hours of data to predict future hours. This leverages the time-series nature of the dataset to augment the training process.


## Steps for Both Approaches

For both approaches, we'll perform the following steps:

* Apply relevant preprocessing steps to prepare the data.
* Run LazyPredict on a large subset of the features (up to 300 columns).
* Define the most important features using SHAP (SHapley Additive exPlanations) or other feature importance methods.
* Identify the most promising models running LazyPredict again, but now with the most important features.
* Perform hyperparameter optimization on the most promising models using SciKit-Optimize for fine-tuning.
* Combine the best-performing models into **ensembles** to enhance prediction accuracy.
* Use the final models to predict blood glucose levels on the test data.
