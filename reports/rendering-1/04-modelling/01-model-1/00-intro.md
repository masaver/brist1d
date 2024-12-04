# Model 1

## Overview
Our first approach uses only the data contained in `train.csv` to train and tune our models. 
The workflow for this approach includes:

* **3a.1**: Testing multiple regressor models with default parameters using LazyPredict
* **3a.2**: Feature importance based on SHAP values
* **3a.3**: Re-running LazyPredict on the feature subset derived from the SHAP analysis
* **3a.4**: Hyperparameter Tuning on Feature Set, Ensemble Regressor & Kaggle submissions

## Usage of LazyPredict
* We are using LazyPredict nightly in the project.
* Why LazyPredict?
    LazyPredict (LP) is an AutoML tool designed for quick testing of multiple estimators (classifiers or regressors) from Scikit-learn. Under the hood, LP defines a list of models to test. It simplifies initial model selection by providing performance metrics for a variety of models without requiring parameter tuning.
* Challenges LazyPredict:   
    In case of Regression, two models from the default list took an unusually long time to process and even caused Jupyter kernels to crash. To circumvent this, the list of regressors was modified to exclude these problematic models.
* Customizing LazyPredict:
To streamline the process and customize the list of regressors tested, a helper function get_lazy_regressor() was created. This function can be imported from ``LazyPredict.py`` in the ``src/helpers`` folder. It provides a convenient instantiation of ``LazyRegressor()`` with a tailored list of regressors.

