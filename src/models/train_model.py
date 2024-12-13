import sys

from src.models.model_1.model.model import Model as Model1
from src.models.model_2.model.model import Model as Model2
from src.models.model_1.data.load_data import load_data as load_data_1
from src.models.model_2.data.load_data import load_data as load_data_2


def train_model_1():
    print('Training model...')
    model = Model1(load=False)

    print('Load and transform train data...')
    X_train, y_train, _ , _ = load_data_1()

    print('Fitting model...')
    model.fit(X_train, y_train)

    print('Saving model...')
    model.save()

    print('Model trained and saved.')

    return model


def train_model_2():
    print('Training model...')
    model = Model2(load=False)

    print('Load and transform train data...')
    X_train, y_train, _ , _ = load_data_2()

    print('Fitting model...')
    model.fit(X_train, y_train)

    print('Saving model...')
    model.save()

    print('Model trained and saved.')

    return model


def train_model(arg: str):
    if arg == '1':
        print('Model 1 selected')
        return train_model_1()
    elif arg == '2':
        print('Model 2 selected')
        return train_model_2()
    else:
        print('Invalid model number. Please use 1 or 2.')
        return None


if __name__ == '__main__':
    default = '2'
    model_to_train = len(sys.argv) > 1 and sys.argv[1] or default
    train_model(model_to_train)
