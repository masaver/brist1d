# Model 1 - **`train.csv`** data only

## Overview
Our first approach uses only the data contained in `train.csv` to train and tune our models. 
The workflow for this approach includes:

* Testing multiple regressor models with default parameters using LazyPredict
* Feature importance based on SHAP values
* Re-running LazyPredict on the feature subset derived from the SHAP analysis
* Hyperparameter Tuning on Feature Set, Ensemble Regressor & Kaggle submissions

## Usage of LazyPredict
* We are using LazyPredict nightly in the project.
* Why LazyPredict?
    LazyPredict (LP) is an AutoML tool designed for quick testing of multiple estimators (classifiers or regressors) from Scikit-learn. Under the hood, LP defines a list of models to test. It simplifies initial model selection by providing performance metrics for a variety of models without requiring parameter tuning.
* Challenges LazyPredict:   
    In case of Regression, two models from the default list took an unusually long time to process and even caused Jupyter kernels to crash. To circumvent this, the list of regressors was modified to exclude these problematic models.
* Customizing LazyPredict:
To streamline the process and customize the list of regressors tested, a helper function get_lazy_regressor() was created. This function can be imported from ``LazyPredict.py`` in the ``src/helpers`` folder. It provides a convenient instantiation of ``LazyRegressor()`` with a tailored list of regressors.

## Feature Importance via SHAP (SHapley Additive exPlanations) Analysis
* SHAP (SHapley Additive exPlanations) is a framework for interpreting machine learning model predictions by attributing the influence of each input feature to the output. SHAP analysis is especially valuable in improving model transparency, identifying key features, and building trust in machine learning applications. It is widely used in critical fields like healthcare, finance, and regulatory environments where interpretability is essential.
  
* **Key Concepts**:
  * **Baseline Prediction**: The average prediction when no features are included.
  * **Feature Contribution**: Each SHAP value represents a feature's contribution to moving the model's prediction away from the baseline.
  
* **Advantages**:
  * **Model-Agnostic**: Can be applied to any machine learning model.
  * **Fair Attribution**: Ensures features are credited proportionally to their impact.
  * **Local and Global Interpretability**: Explains individual predictions and overall model behavior.












