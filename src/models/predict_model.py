import pandas as pd
from src.models.model.model import Model
from src.models.data.load_data import load_data


def predict_model():
    print('Load or create model...')
    model = Model(load=True)

    print('Load test data...')
    _, _, X_test = load_data()

    print('Predicting...')
    y_pred = model.predict(X_test)

    print('Prediction:', y_pred)
    return y_pred


if __name__ == '__main__':
    predict_model()
