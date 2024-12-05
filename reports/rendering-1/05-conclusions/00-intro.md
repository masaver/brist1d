# Conclusions

## Summary

In this report, we have presented two models to predict Blood Glucose levels one hour ahead using the previous six hours of participant data. We have explored the data,
preprocessed it, and engineered features to improve the model's performance. We have also implemented a custom cross-validation strategy to evaluate the model's performance and
optimize its hyperparameters.

Two approaches of model training have been presented in this report. The first approach used the training data as is, while the second approach augmented the training data by
calculating augmented data from the test data. The augmented data was used to train the model, which improved the model's performance by providing additional training examples.

Our best model achieved a Kaggle Score of 2.45 on the public leaderboard, which is a significant improvement over the baseline model. The model's performance was further validated
using a custom cross-validation strategy, which showed consistent results across different folds.

## Challenges

Some challenges were encountered during the development of the model, which we addressed through various strategies.

### Imbalanced Training and Test Data

The training and test data were imbalanced based on the observed patients. We had no training data for some patients, which made it challenging to predict their blood glucose
with only the training data. We addressed this issue by augmenting the training data from the test dataset.

### Kaggle RMSE and Local RMSE Discrepancy

We found it very challenging in our model development to get the local RMSE to match the Kaggle RMSE. We have tried different strategies to address this issue, including feature
selection, hyperparameter tuning, and cross-validation. We have different hypotheses on why this discrepancy exists, which have to be proven in future work.

* The number of features is too high and introduces noise into the model
* The model is overfitting the training data (Trees too deep, KNeighbors too low) and not generalizing well to the test data on kaggle
* The model is not able to capture the underlying patterns in the data

## Results

The results of the challenge were very promising, with our best model achieving a Kaggle Score of 2.3472 on the public leaderboard.
This means that our model was able to predict 20% of the test data with this RMSE.

![public-leaderboard.png](../../figures/kaggle-public-leaderboard.png)

After the official competition ended, all selected models were evaluated on the 80% of the test data that was not used in the public leaderboard. Our model achieved a RMSE of
2.4391 and was ranked 10th in the competition.

## Final Thoughts

As a team, we have learned a lot from this project, including data preprocessing, feature engineering, model development, and evaluation. We have also learned how to work
collaboratively on a project, communicate effectively, and solve problems as a team. We applied various machine learning techniques and stacked these into a robust model to predict
Blood Glucose levels one hour ahead.

Our solution was sored in the top 1% of the Kaggle public Leaderboard, which is a great achievement for us. We are proud of our work and the effort we put into this project. We
believe that the skills we have developed during this project will be valuable in our future careers as data scientists and machine learning engineers.
