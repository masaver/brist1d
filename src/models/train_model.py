from src.models.model.model import Model
from src.models.data.load_data import load_data


def train_model():
    print('Training model...')
    model = Model(load=False)

    print('Load and transform train data...')
    X_train, y_train, _ = load_data()

    print('Fitting model...')
    model.fit(X_train, y_train)

    print('Saving model...')
    model.save()

    print('Model trained and saved.')


if __name__ == '__main__':
    train_model()
