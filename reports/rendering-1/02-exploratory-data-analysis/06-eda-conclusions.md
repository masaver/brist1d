# Exploratory Data analysis conclusions

Our Exploratory data analysis  has so far revealed that:
* Only a minor anomaly was detected for p12, where we found two value
* Only one patient (p11) shows small inconsistencies across continuous  rows in the raw data
* Feature columns in tha raw data have moderate fractions of NAs. However, columns related to **activity** and **carbs** have a substantial ammount of NAs (>95%)
* With minor exceptions, the distribution of features and target variable is comparable across patients
* The distribution for all variables are right-skewed. 
* Most features ( whole time series or lag features ) are poorly correlated to bg+1:00


## Business relevant insights

In theory, the main advantage of the dataset we are working with is that it provides high-throughput measurements for seveal variables that are know to affect blood glucose. However, directly using some of the information provided will be challenging. More concretely:

* Using the columns related to **activity** and **carbs** will more difficult, if not impractica, given that they are almost all NAs. The **activity**  also contain information for multiple types of activity. So using it for modelling would requiere some kind of encoding
  
* Except for prior blood glucose (**bg**) levels, all variables show a very poor correlation with the outcome variable, so it's entirely possible that a good performingl model can already be achieved with with prior blood glucose levels alone. This can mean  that a good product/technology might requiere extra complicated devices and measuremens.

## Planned steps for upcomming analysis

### Preprocessing & Preliminary Modelling:
* Impute the raw data with the columns median ( recall that most distributions are skewed ) and Run LazyPredict
* Impute the processed data ( long format ) with column meadian or time series interpolation and run LazyPredict


### NOTE/Ref:
(Lazy Predict)[https://github.com/shankarpandala/lazypredict] helps build a lot of basic models ( for Classification or Regression) without much code and helps understand which models works better without any parameter tuning.