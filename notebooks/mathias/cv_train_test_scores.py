from sklearn.metrics import mean_squared_error
from sklearn.metrics import make_scorer, mean_squared_error
rmse_scorer = make_scorer(mean_squared_error, greater_is_better=False, squared=False)

from sklearn.model_selection import cross_validate
import xgboost as xgb

# Initialize the model
model = xgb.XGBRegressor()
#model = xgb_tuner.get_params()

# Perform cross-validation, specifying both train and test scores
cv_results = cross_validate(
    model, Xs, y, cv=5,
    scoring = rmse_scorer , 
    return_train_score=True
)

# Extract train and test scores
train_scores = cv_results['train_score']
test_scores = cv_results['test_score']

# Display the results
print("Train scores for each fold:", -1*train_scores)
print("Test scores for each fold:", -1*test_scores)
print("Average train score:", -1*train_scores.mean())
print("Average test score:", -1*test_scores.mean())