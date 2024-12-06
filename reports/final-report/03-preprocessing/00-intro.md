# Preprocessing

The objective of this section is to show and to explain the various preprocessing steps and feature engineering techniques that have been applied by our team in the context of the Kaggle competition. These steps are essential to transform the raw data into a format suitable for modelling and to maximize the performance of predictive models.

## Preprocessing steps

This section outlines the key preprocessing steps, carried out by our team, which, in our opinion, were necessary to properly prepare the dataset for different types of modelling.

We developed a bunch of custom [transformers](https://scikit-learn.org/stable/data_transforms.html) tailored to the structure and characteristics of our dataset. These transformers are used to preprocess the data both during training and prediction phases, ensuring consistency and efficiency.

* Objectives of custom transformers:
    * Handle missing values and anomalies.
    * Format the data to align with the requirements of time series.
    * Generate additional features that might improve the model's predictive performance.

<!--
see: [Preprocessing steps](01-preprocessing-steps.md)
-->

