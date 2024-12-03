import tensorflow as tf
from scikeras.wrappers import KerasRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LassoLarsIC
from sklearn.neighbors import KNeighborsRegressor
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input, Activation
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from xgboost import XGBRegressor


def rmse(y_true, y_pred):
    return tf.sqrt(tf.reduce_mean(tf.square(y_pred - y_true)))


def create_dnn_model(input_dimension: int):
    dnn = Sequential([
        Input(shape=(input_dimension,)),
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),

        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),

        Dense(32, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),

        Dense(1, activation='linear')
    ])

    dnn.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='mse',
        metrics=[rmse]
    )

    return dnn


weights_filename = 'model.weights.h5'
def get_keras_regressor(X_train, y_train):
    pretrained_dnn = create_dnn_model(X_train.shape[1])
    pretrained_dnn.fit(
        X_train,
        y_train,
        epochs=100,
        verbose=2,
        validation_split=0.2,
        batch_size=32,
        callbacks=[EarlyStopping(monitor='val_rmse', patience=10, restore_best_weights=True, mode= 'min')]
    )

    pretrained_dnn.save_weights(weights_filename)

    def model_with_pretrained_weights():
        dnn = create_dnn_model(X_train.shape[1])
        dnn.load_weights(weights_filename)  # Load the pre-trained weights

        # Freeze all layers except the last one (optional)
        for layer in dnn.layers:
            layer.trainable = False
        # Unfreeze the output layer if you want to fine-tune it
        # dnn.layers[-1].trainable = True

        dnn.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=[rmse]
        )
        return dnn

    return KerasRegressor(
        model=model_with_pretrained_weights,
        epochs=1,
        verbose=2
    )


def get_hgb_regressor():
    return HistGradientBoostingRegressor(max_iter=200, max_depth=5, learning_rate=0.01)


def get_lasso_lars_ic_regressor():
    return LassoLarsIC(criterion='bic', max_iter=10000)


def get_knn_regressor():
    return KNeighborsRegressor(n_neighbors=5)


def get_xgb_regressor():
    return XGBRegressor(objective='reg:squarederror', random_state=42, n_estimators=500, max_depth=5, learning_rate=0.1)
