import os
import pandas as pd


def load_data():
    train_data_file = os.path.join('..', '..', '..', '..', 'data', 'raw', 'train.csv')
    train_data = pd.read_csv(train_data_file, index_col=0, low_memory=False)

    additional_train_data_file = os.path.join('..', '..', '..', '..', 'data', 'interim', 'all_test_2h.csv')
    additional_train_data = pd.read_csv(additional_train_data_file, index_col=0, low_memory=False)

    train_data = pd.concat([train_data, additional_train_data], axis=0)

    # do not train with patients that are not have to be predicted
    test_data_file = os.path.join('..', '..', '..', '..', 'data', 'raw', 'test.csv')
    test_data = pd.read_csv(test_data_file, index_col=0, low_memory=False)

    unique_patients = test_data['p_num'].unique()
    train_data = train_data[train_data['p_num'].isin(unique_patients)]
    test_data = test_data[test_data['p_num'].isin(unique_patients)]

    return train_data, test_data


def load_data_selected_features():
    train_data_file = os.path.join('..', '..', '..', '..', 'data', 'raw', 'train.csv')
    train_data = pd.read_csv(train_data_file, index_col=0, low_memory=False)

    additional_train_data_file = os.path.join('..', '..', '..', '..', 'data', 'interim', 'all_test_1h.csv')
    additional_train_data = pd.read_csv(additional_train_data_file, index_col=0, low_memory=False)

    # do not train with patients that are not have to be predicted
    test_data_file = os.path.join('..', '..', '..', '..', 'data', 'raw', 'test.csv')
    test_data = pd.read_csv(test_data_file, index_col=0, low_memory=False)

    unique_patients = test_data['p_num'].unique()
    train_data = train_data[train_data['p_num'].isin(unique_patients)]
    additional_train_data = additional_train_data[additional_train_data['p_num'].isin(unique_patients)]
    test_data = test_data[test_data['p_num'].isin(unique_patients)]

    return train_data, additional_train_data, test_data
