from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import root_mean_squared_error, PredictionErrorDisplay, r2_score
from sklearn.model_selection import train_test_split
from skopt import BayesSearchCV
import pandas as pd

from src.features.helpers import CustomSplitter


def get_date_time_now():
    return datetime.now().strftime('%H:%M:%S')


def print_conditionally(message, verbose=True):
    if verbose:
        print(f'{get_date_time_now()} - {message}')


def show_stats(y_true, y_pred):
    print(f'RMSE: {root_mean_squared_error(y_true, y_pred)}')
    print(f'R2 Score: {r2_score(y_true, y_pred)}')

    fig, axs = plt.subplots(ncols=2, figsize=(8, 4))
    PredictionErrorDisplay.from_predictions(
        y_true=y_true,
        y_pred=y_pred,
        kind="actual_vs_predicted",
        subsample=100,
        ax=axs[0],
        random_state=0,
    )
    axs[0].set_title("Actual vs. Predicted values")
    PredictionErrorDisplay.from_predictions(
        y_true=y_true,
        y_pred=y_pred,
        kind="residual_vs_predicted",
        subsample=100,
        ax=axs[1],
        random_state=0,
    )
    axs[1].set_title("Residuals vs. Predicted Values")
    fig.suptitle("Plotting cross-validated predictions")
    plt.tight_layout()
    plt.show()


def tune_hyperparameters(model, param_space, X_train, y_train, X_additional_train, y_additional_train, verbose=True, num_iter=50, n_splits=5):
    print_conditionally(f'Start tuning {model.__class__.__name__}', verbose)
    print_conditionally(f'Parameters: {param_space}', verbose)

    X_all_train = pd.concat([X_train, X_additional_train])
    y_all_train = pd.concat([y_train, y_additional_train])

    splitter = CustomSplitter(test_size=0.2, n_splits=n_splits, random_state=42)
    splitter.fit(X_all_train, groups=[0] * len(X_train) + [1] * len(X_additional_train))

    _, X_eval, _, y_eval = train_test_split(X_additional_train, y_additional_train, test_size=0.2, random_state=42, shuffle=True)
    fit_params = {
        "eval_set": [(X_eval, y_eval)],
        "early_stopping_rounds": 50,
        "verbose": True
    }

    np.int = int
    search_cv = BayesSearchCV(
        estimator=model,
        search_spaces=param_space,
        n_iter=num_iter,
        scoring='neg_mean_squared_error',
        cv=splitter,
        n_jobs=-1,
        random_state=42,
        verbose=1 if verbose else 0,
        fit_params=fit_params
    )

    print_conditionally(f'Fitting the model', verbose)
    search_cv.fit(X_all_train, y_all_train)

    print_conditionally(f'Best hyperparameters found.', verbose)
    print_conditionally(search_cv.best_params_, verbose)

    if verbose:
        y_all_pred = search_cv.predict(X_all_train)
        show_stats(y_all_train, y_all_pred)

    return search_cv.best_estimator_, search_cv.best_params_
