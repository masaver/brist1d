# Modelling based on train data

Our first approach was  to take  only the data contained in `train.csv` to train/tune our models. 
The overall workflow of this section was:
* **3a.1**: Testing Multiple regressor models with default parameters via LazyPredict
* **3a.2**: Feature importance based on SHAP values
* **3a.3**: LazyPredict Modelling on feature subset
* **3a.4**: Model Tuning on Feature Set, Ensemble Regressor & Kaggle submissions

# Usage of LazyPredict
* We are using lazyPredict nightly in the project.
* LazyPredict (LP) is a type of AutoML tool, that allows you to test multiple extimators( Classifiers or Regressors ) from sci-kit learn. Under the hood, LP defines a list of models to test. In the case of Regression, two of the models were talking a relly loong time and even crashed the jupyter kernels. To circumvent that, modifiedd slightly the list of regressors to test.
* To conveniently instantiate a `LazyRegressor()` and to custumize the list of individual regressors tested, you can import `get_lazy_regressor()` from `LazyPredict.py` in the helpers folder (see `src/helpers` ). 