from src.models.model_1.model.model import Model as Model1
from src.models.model_2.model.model import Model as Model2
from src.models.model_1.data.load_data import load_data as load_data_1
from src.models.model_2.data.load_data import load_data as load_data_2
import sys


def predict_model_1():
    print('Load or create model...')
    model = Model1(load=True)

    print('Load test data...')
    _, _, X_test, _ = load_data_1()

    print('Predicting...')
    y_pred = model.predict(X_test)

    print('Prediction:', y_pred)
    return y_pred


def predict_model_2():
    print('Load or create model...')
    model = Model2(load=True)

    print('Load test data...')
    _, _, X_test = load_data_2()

    print('Predicting...')
    y_pred = model.predict(X_test)

    print('Prediction:', y_pred)
    return y_pred


def predict_model(arg: str):
    if arg == '1':
        print('Model 1 selected')
        return predict_model_1()
    elif arg == '2':
        print('Model 2 selected')
        return predict_model_2()
    else:
        print('Invalid model number. Please use 1 or 2.')
        return None


if __name__ == '__main__':
    default = '2'
    model_to_predict = len(sys.argv) > 1 and sys.argv[1] or default
    predict_model(model_to_predict)
