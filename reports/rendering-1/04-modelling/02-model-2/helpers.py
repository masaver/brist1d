import json
from dataclasses import dataclass
from datetime import datetime
from typing import Callable

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from jedi.inference.gradual.typing import TypedDict
from sklearn.ensemble import StackingRegressor
from sklearn.model_selection import ShuffleSplit, LeaveOneGroupOut
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, mean_squared_error, r2_score


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


def calculate_stacking_regressor_performance(model: StackingRegressor, X_train, y_train, X_additional_train, y_additional_train, verbose=True, n_splits=5, groups=None):
    print_conditionally(f'Start training', verbose=verbose)

    models = model.estimators + [('final_estimator', model.final_estimator), ('stacking_regressor', model)]
    model_scores = []

    for model_name, model in models:
        print_conditionally(f'Calculate performance for {model_name}', verbose=verbose)

        splitter = ShuffleSplit(n_splits=n_splits, test_size=0.2, random_state=42)
        if groups is not None:
            splitter = LeaveOneGroupOut()
            print(f'Number of splits: {splitter.get_n_splits(groups=groups)}')

        print_conditionally(f'Selected splitter: {splitter}', verbose=verbose)

        score = {}
        split_index = 0
        for train_index, test_index in splitter.split(X_additional_train, groups=groups):
            split_index += 1
            print_conditionally(f'Split {split_index} - Model: {model_name}', verbose=verbose)

            X_train_split, X_test_split = X_additional_train.iloc[train_index], X_additional_train.iloc[test_index]
            y_train_split, y_test_split = y_additional_train.iloc[train_index], y_additional_train.iloc[test_index]

            print_conditionally(f'Fitting the model', verbose=verbose)
            model.fit(pd.concat([X_train, X_train_split]), pd.concat([y_train, y_train_split]))

            print_conditionally(f'Predicting', verbose=verbose)
            y_pred = model.predict(X_test_split)

            print_conditionally(f'Calculating scores', verbose=verbose)

            r_squared = r2_score(y_test_split, y_pred)
            rmse = root_mean_squared_error(y_test_split, y_pred)
            mae = mean_absolute_error(y_test_split, y_pred)
            mse = mean_squared_error(y_test_split, y_pred)

            score['r_squared'] = score.get('r_squared', []) + [r_squared]
            score['rmse'] = score.get('rmse', []) + [rmse]
            score['mae'] = score.get('mae', []) + [mae]
            score['mse'] = score.get('mse', []) + [mse]

            print_conditionally(f'R^2: {r_squared}, RMSE: {rmse}, MAE: {mae}, MSE: {mse}', verbose=verbose)

        print_conditionally(f'Training finished', verbose=verbose)

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


def calculate_dnn_performance(create_model_fn: Callable, X_train, y_train, X_additional_train, y_additional_train, verbose=True, n_splits=5, epochs=50, groups=None,
                              callbacks=None):
    print_conditionally(f'Start training DNN', verbose)

    if verbose:
        create_model_fn(X_train.shape[1]).summary()

    splitter = ShuffleSplit(n_splits=n_splits, test_size=0.2, random_state=42)
    if groups is not None:
        splitter = LeaveOneGroupOut()

    print_conditionally(f'Selected splitter: {splitter}', verbose)

    score = {}
    histories = []
    split_index = 0
    for train_index, test_index in splitter.split(X=X_additional_train, groups=groups):
        split_index += 1
        print_conditionally(f'Split {split_index}/{n_splits}', verbose)
        X_train_split, X_test_split = X_additional_train.iloc[train_index], X_additional_train.iloc[test_index]
        y_train_split, y_test_split = y_additional_train.iloc[train_index], y_additional_train.iloc[test_index]

        model = create_model_fn(X_train.shape[1])

        model.fit(
            pd.concat([X_train, X_train_split]),
            pd.concat([y_train, y_train_split]),
            validation_data=(X_test_split, y_test_split),
            epochs=epochs,
            callbacks=(callbacks or []),
            verbose=2 if verbose else 0
        )

        histories.append(model.history.history)
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
    return model_score, histories


def get_rmse_boxplot_chart(scores: list[ModelScore] | ModelScore):
    if isinstance(scores, ModelScore):
        scores = [scores]

    final_estimator_score = scores[-1]
    print(f'Final estimator RMSE: {final_estimator_score.rmse["mean"]}')
    print(f'Final estimator R2: {final_estimator_score.r_squared["mean"]}')
    print(f'Final estimator MSE: {final_estimator_score.mse["mean"]}')
    print(f'Final estimator MAE: {final_estimator_score.mae["mean"]}')
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


def get_history_line_chart(histories: list[dict]):
    final_rmse = np.round(np.mean([history['val_rmse'][-1] for history in histories]), 4)
    plt.figure(figsize=(10, 5))
    plt.plot(np.mean([history['rmse'] for history in histories], axis=0), label='Training RMSE', color='b')
    plt.plot(np.mean([history['val_rmse'] for history in histories], axis=0), label='Test RMSE', color='r')
    for history in histories:
        plt.plot(history['val_rmse'], linestyle='--', color='r', alpha=0.3)

    # put a vertical line at epoch with minimum mean validation RMSE
    mean_rmse = np.mean([history['val_rmse'] for history in histories], axis=0)
    min_rmse_epoch = np.argmin(mean_rmse)
    plt.axvline(x=min_rmse_epoch, color='g', linestyle='--', label=f'Min RMSE epoch: {min_rmse_epoch}')

    plt.legend()
    plt.ylabel('RMSE')
    plt.xlabel('Epoch')
    plt.gca().xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    plt.title(f'RMSE over epochs (Final: {final_rmse})')
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


def plot_feature_importance_chart(feature_importances: pd.DataFrame):
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    fig = plt.figure(figsize=(12, 8))
    gs = gridspec.GridSpec(3, 2, height_ratios=[1, 1, 1])

    # First plot: spans the first row (2 columns)
    ax1 = fig.add_subplot(gs[0, :])
    feature_importances.loc[feature_importances.index.str.startswith('bg')].plot(kind='bar', ax=ax1, stacked=True)
    ax1.set_title('Feature importance blood glucose')
    ax1.set_xlabel('Feature')
    ax1.set_xticklabels([item.get_text()[-5:] for item in ax1.get_xticklabels()], rotation=45, horizontalalignment='right')
    ax1.set_ylabel('Importance')
    ax1.legend()

    ax2 = fig.add_subplot(gs[1, 0])
    feature_importances.loc[feature_importances.index.str.startswith('insulin')].plot(kind='bar', ax=ax2, stacked=True)
    ax2.set_title('Feature importance insulin')
    ax2.set_xlabel('Feature')
    ax2.set_xticklabels([item.get_text()[-5:] for item in ax2.get_xticklabels()], rotation=45, horizontalalignment='right')
    ax2.set_ylabel('Importance')
    ax2.legend()

    # Second row, right: Share y-axis with the previous plot
    ax3 = fig.add_subplot(gs[1, 1], sharey=ax2)
    feature_importances.loc[feature_importances.index.str.startswith('hr')].plot(kind='bar', ax=ax3, stacked=True)
    ax3.set_title('Feature importance heart rate')
    ax3.set_xlabel('Feature')
    ax3.set_xticklabels([item.get_text()[-5:] for item in ax3.get_xticklabels()], rotation=45, horizontalalignment='right')
    ax3.set_ylabel('Importance')
    ax3.legend()

    # Third row, left: Share y-axis with the next plot
    ax4 = fig.add_subplot(gs[2, 0])
    feature_importances.loc[feature_importances.index.str.startswith('steps')].plot(kind='bar', ax=ax4, stacked=True)
    ax4.set_title('Feature importance steps')
    ax4.set_xlabel('Feature')
    ax4.set_xticklabels([item.get_text()[-5:] for item in ax4.get_xticklabels()], rotation=45, horizontalalignment='right')
    ax4.set_ylabel('Importance')
    ax4.legend()

    # Third row, right: Share y-axis with the previous plot
    ax5 = fig.add_subplot(gs[2, 1], sharey=ax4)
    feature_importances.loc[feature_importances.index.str.startswith('cals')].plot(kind='bar', ax=ax5, stacked=True)
    ax5.set_title('Feature importance calories')
    ax5.set_xlabel('Feature')
    ax5.set_xticklabels([item.get_text()[-5:] for item in ax5.get_xticklabels()], rotation=45, horizontalalignment='right')
    ax5.set_ylabel('Importance')
    ax5.legend()

    # Adjust layout
    plt.tight_layout()
    plt.show()
