from src.features.helpers.load_data import load_data as common_load_data


def load_data():
    train, augmented, test = common_load_data(time_frame='2_00h', only_patients_in_test=True)

    return train, augmented, test


def load_data_selected_features():
    train, augmented, test = common_load_data(time_frame='1_00h', only_patients_in_test=True)

    return train, augmented, test
