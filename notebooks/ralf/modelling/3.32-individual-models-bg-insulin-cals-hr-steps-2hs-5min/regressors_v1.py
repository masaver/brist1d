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


def get_keras_regressor(p_num, X_train, X_val, y_train, y_val):
    pretrained_dnn = create_dnn_model(X_train.shape[1])
    pretrained_dnn.fit(
        X_train,
        y_train,
        epochs=100,
        verbose=2,
        validation_data=(X_val, y_val),
        validation_split=0.2,
        batch_size=32,
        callbacks=[
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        ]
    )

    pretrained_dnn.save_weights(f'{p_num}.weights.h5')

    def model_with_pretrained_weights():
        dnn = create_dnn_model(X_train.shape[1])
        dnn.load_weights(f'{p_num}.weights.h5')  # Load the pre-trained weights

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
    return HistGradientBoostingRegressor(
        l2_regularization=0.0001,
        learning_rate=0.1,
        max_depth=11,
        max_iter=1500,
        min_samples_leaf=10,
        n_iter_no_change=22,
        random_state=42,
    )


def get_lasso_lars_ic_regressor():
    return LassoLarsIC(
        criterion='aic',
        eps=0.03922948513965659,
        max_iter=1944,
        noise_variance=5.4116687755186035e-05,
        positive=False,
    )


def get_knn_regressor():
    return KNeighborsRegressor(
        leaf_size=30,
        metric='minkowski',
        n_neighbors=7,
        p=2,
        weights='uniform'
    )


def get_xgb_regressor():
    return XGBRegressor(
        n_estimators=1000,
        max_depth=6,
        learning_rate=0.05,
        min_child_weight=6,
        colsample_bytree=0.8,
        subsample=0.8,
        gamma=1,
        reg_lambda=10.0,
        reg_alpha=1.0,
        objective='reg:squarederror',
        random_state=42,
    )
