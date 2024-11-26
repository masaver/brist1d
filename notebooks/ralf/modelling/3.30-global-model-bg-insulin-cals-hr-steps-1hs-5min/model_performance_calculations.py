import json
from dataclasses import dataclass
from datetime import datetime

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from jedi.inference.gradual.typing import TypedDict
from sklearn.ensemble import StackingRegressor
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping


class ModelScoreParameter(TypedDict):
    mean: float
    std: float
    min: float
    max: float
    values: list[float]


@dataclass
class ModelScore:
    name: str
    r_squared: ModelScoreParameter | None
    rmse: ModelScoreParameter | None
    mae: ModelScoreParameter | None
    mse: ModelScoreParameter | None

    def __init__(self, name: str = '', r_squared: ModelScoreParameter | None = None, rmse: ModelScoreParameter | None = None, mae: ModelScoreParameter | None = None,
                 mse: ModelScoreParameter | None = None):
        self.name = name
        self.r_squared = r_squared
        self.rmse = rmse
        self.mae = mae
        self.mse = mse

    @classmethod
    def from_dict(cls, data):
        return cls(
            name=data.get('name', ''),
            r_squared=data.get('r_squared', None),
            rmse=data.get('rmse', None),
            mae=data.get('mae', None),
            mse=data.get('mse', None)
        )

    def __getitem__(self, key):
        return getattr(self, key)

    def to_dict(self):
        return {
            'name': self.name,
            'r_squared': self.r_squared,
            'rmse': self.rmse,
            'mae': self.mae,
            'mse': self.mse
        }


def get_date_time_now():
    return datetime.now().strftime('%H:%M:%S')


def print_conditionally(message, verbose=True):
    if verbose:
        print(f'{get_date_time_now()} - {message}')


def calculate_stacking_regressor_performance(model: StackingRegressor, X_train, y_train, X_additional_train, y_additional_train, verbose=True, n_splits=5):
    print_conditionally(f'{get_date_time_now()} - Start training', verbose)

    models = model.estimators + [('final_estimator', model.final_estimator), ('stacking_regressor', model)]
    model_scores = []

    for model_name, model in models:
        print_conditionally(f'{get_date_time_now()} - Calculate performance for {model_name}', verbose)

        score = {}
        split_index = 0
        print_conditionally(f'{get_date_time_now()} - Splitting the additional train data with ShuffleSplit')
        splitter = ShuffleSplit(n_splits=n_splits, test_size=0.2, random_state=42)
        for train_index, test_index in splitter.split(X_additional_train):
            split_index += 1
            print_conditionally(f'{get_date_time_now()} - Split {split_index} - Model: {model_name}', verbose)

            X_train_split, X_test_split = X_additional_train.iloc[train_index], X_additional_train.iloc[test_index]
            y_train_split, y_test_split = y_additional_train.iloc[train_index], y_additional_train.iloc[test_index]

            print_conditionally(f'{get_date_time_now()} - Fitting the model')
            model.fit(pd.concat([X_train, X_train_split]), pd.concat([y_train, y_train_split]))

            print_conditionally(f'{get_date_time_now()} - Predicting')
            y_pred = model.predict(X_test_split)

            print_conditionally(f'{get_date_time_now()} - Calculating scores')
            r_squared = model.score(X_test_split, y_test_split)
            rmse = root_mean_squared_error(y_test_split, y_pred)
            mae = mean_absolute_error(y_test_split, y_pred)
            mse = mean_squared_error(y_test_split, y_pred)

            score['r_squared'] = score.get('r_squared', []) + [r_squared]
            score['rmse'] = score.get('rmse', []) + [rmse]
            score['mae'] = score.get('mae', []) + [mae]
            score['mse'] = score.get('mse', []) + [mse]

            print_conditionally(f'{get_date_time_now()} - R^2: {r_squared}, RMSE: {rmse}, MAE: {mae}, MSE: {mse}', verbose)

        print_conditionally(f'{get_date_time_now()} - Training finished', verbose)

        score = {k: {
            'mean': np.mean(v),
            'std': np.std(v),
            'min': np.min(v),
            'max': np.max(v),
            'values': v
        } for k, v in score.items()}

        model_score = ModelScore(model_name)

        model_score.r_squared = score['r_squared']
        model_score.rmse = score['rmse']
        model_score.mae = score['mae']
        model_score.mse = score['mse']
        model_scores.append(model_score)

    return model_scores


def calculate_dnn_performance(model: Sequential, X_train, y_train, X_additional_train, y_additional_train, verbose=True, n_splits=5, epochs=50):
    print_conditionally(f'Start training DNN', verbose)

    if verbose:
        model.summary()

    score = {}
    splitter = ShuffleSplit(n_splits=n_splits, test_size=0.2, random_state=42)
    split_index = 0
    for train_index, test_index in splitter.split(X_additional_train):
        split_index += 1
        print_conditionally(f'Split {split_index}/{n_splits}', verbose)
        X_train_split, X_test_split = X_additional_train.iloc[train_index], X_additional_train.iloc[test_index]
        y_train_split, y_test_split = y_additional_train.iloc[train_index], y_additional_train.iloc[test_index]

        early_stop = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )

        model.fit(
            pd.concat([X_train, X_train_split]),
            pd.concat([y_train, y_train_split]),
            validation_data=(X_test_split, y_test_split),
            epochs=epochs,
            callbacks=[early_stop],
            verbose=2
        )

        y_pred = model.predict(X_test_split)

        r_squared = r2_score(y_test_split, y_pred)
        rmse = root_mean_squared_error(y_test_split, y_pred)
        mae = mean_absolute_error(y_test_split, y_pred)
        mse = mean_squared_error(y_test_split, y_pred)

        score['r_squared'] = score.get('r_squared', []) + [r_squared]
        score['rmse'] = score.get('rmse', []) + [rmse]
        score['mae'] = score.get('mae', []) + [mae]
        score['mse'] = score.get('mse', []) + [mse]

        print_conditionally(f'R^2: {r_squared}, RMSE: {rmse}, MAE: {mae}, MSE: {mse}', verbose)

    print_conditionally('Training finished', verbose)

    score = {k: {
        'mean': np.mean(v),
        'std': np.std(v),
        'min': np.min(v),
        'max': np.max(v),
        'values': v
    } for k, v in score.items()}

    model_score = ModelScore('DNN')
    model_score.r_squared = score['r_squared']
    model_score.rmse = score['rmse']
    model_score.mae = score['mae']
    model_score.mse = score['mse']
    return model_score


def get_rmse_boxplot_chart(scores: list[ModelScore] | ModelScore):
    if isinstance(scores, ModelScore):
        scores = [scores]

    final_estimator_score = scores[-1]
    values = []
    labels = []
    for score in scores:
        values.append(score.rmse['values'])
        labels.append(score.name)
    plt.boxplot(values, labels=labels)
    final_estimator_mean = np.round(final_estimator_score.rmse['mean'], 4)
    plt.xlabel(f'Final estimator RMSE: {final_estimator_mean}')
    plt.ylabel('RMSE')
    plt.title('RMSE values for each estimator and the final estimator')
    return plt


def get_rmse_line_chart(scores: list[ModelScore] | ModelScore):
    if isinstance(scores, ModelScore):
        scores = [scores]

    final_estimator_score = scores[-1]

    for score in scores:
        plt.plot(score.rmse['values'], label=score.name)
        plt.axhline(y=score.rmse['mean'], color='r', linestyle='--', label=f'{score.name} mean')

    final_estimator_mean = np.round(final_estimator_score.rmse['mean'], 4)
    plt.xlabel(f'Final estimator RMSE: {final_estimator_mean}')
    plt.legend()
    plt.ylabel('RMSE')
    plt.title('RMSE values for each estimator and the final estimator by split')
    return plt


def save_performances(model_scores: list[ModelScore], path: str):
    model_scores_dict = [model_score.to_dict() for model_score in model_scores]
    with open(path, 'w') as f:
        json.dump(model_scores_dict, f)


def save_model(model, path):
    joblib.dump(model, path)
