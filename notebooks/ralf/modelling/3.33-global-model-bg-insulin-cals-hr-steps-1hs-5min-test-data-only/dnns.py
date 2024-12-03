import tensorflow as tf
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Add, Concatenate, Dense, Dropout, BatchNormalization, Input, Activation
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam


def rmse(y_true, y_pred):
    return tf.sqrt(tf.reduce_mean(tf.square(y_pred - y_true)))


# MA01 - Simple DNN
def create_model_MA01(input_dimension: int):
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


# MA01I - Increased Regularization
def create_model_MA01I(input_dimension: int):
    dnn = Sequential([
        Input(shape=(input_dimension,)),

        Dense(128, activation='relu', kernel_regularizer=regularizers.l2(1e-4)),
        Dropout(0.3),

        Dense(64, activation='relu', kernel_regularizer=regularizers.l2(1e-4)),
        Dropout(0.3),

        Dense(32, activation='relu', kernel_regularizer=regularizers.l2(1e-4)),
        Dropout(0.3),

        Dense(1, activation='linear')
    ])

    dnn.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='mse',
        metrics=[rmse]
    )

    return dnn


# MA01I2 - Increased Regularization
def create_model_MA01I2(input_dimension: int):
    dnn = Sequential([
        Input(shape=(input_dimension,)),

        Dense(128, kernel_regularizer=regularizers.l2(1e-4)),
        BatchNormalization(),
        Dense(128, activation='relu'),
        Dropout(0.3),

        Dense(64, kernel_regularizer=regularizers.l2(1e-4)),
        BatchNormalization(),
        Dense(64, activation='relu'),
        Dropout(0.3),

        Dense(32, kernel_regularizer=regularizers.l2(1e-4)),
        BatchNormalization(),
        Dense(32, activation='relu'),
        Dropout(0.3),

        Dense(1, activation='linear')
    ])

    dnn.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='mse',
        metrics=[rmse]
    )

    return dnn


# MA02 - Increased Neurons in Dense Layers
def create_model_MA02(input_dimension: int):
    dnn = Sequential([
        # Input layer
        Input(shape=(input_dimension,)),

        # First dense block
        Dense(512, activation='relu'),  # Increased neurons
        BatchNormalization(),
        Dropout(0.3),  # Slightly reduced dropout for better capacity

        # Second dense block
        Dense(256, activation='relu'),  # Increased neurons
        BatchNormalization(),
        Dropout(0.3),

        # Third dense block
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),

        # Fourth dense block (new)
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),

        # Output layer
        Dense(1, activation='linear')  # Regression output
    ])

    # Compile the model
    dnn.compile(
        optimizer=Adam(learning_rate=0.001),  # Retained learning rate
        loss='mse',  # Mean Squared Error for regression
        metrics=[rmse]  # Custom RMSE metric
    )

    return dnn


# MA03 - Adjusted Layer Sizes with More Nonlinearity
def create_model_MA03(input_dimension: int):
    dnn = Sequential([
        Input(shape=(input_dimension,)),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.4),

        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.4),

        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.4),

        Dense(1, activation='linear')
    ])

    dnn.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='mse',
        metrics=[rmse]
    )

    return dnn


# MA04 - Switched to Exponential Linear Unit (ELU) activation function
def create_model_MA04(input_dimension: int):
    dnn = Sequential([
        Input(shape=(input_dimension,)),
        Dense(128, activation='elu'),
        BatchNormalization(),
        Dropout(0.3),

        Dense(64, activation='elu'),
        BatchNormalization(),
        Dropout(0.3),

        Dense(32, activation='elu'),
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


# MA05 - Residual Connections (Inspired by ResNet)
def create_model_MA05(input_dimension: int):
    def residual_block(input_layer, units):
        # First dense layer
        dense1 = Dense(units, activation='relu')(input_layer)
        batchnorm1 = BatchNormalization()(dense1)
        # Second dense layer
        dense2 = Dense(units, activation='relu')(batchnorm1)
        batchnorm2 = BatchNormalization()(dense2)
        # Add projection if shapes don't match
        if input_layer.shape[-1] != units:
            input_layer = Dense(units)(input_layer)  # Project input to match dimensions
        # Add the skip connection
        return Add()([input_layer, batchnorm2])

    input_layer = Input(shape=(input_dimension,))
    residual1 = residual_block(input_layer, 128)
    residual2 = residual_block(residual1, 64)
    residual3 = residual_block(residual2, 32)

    output = Dense(1, activation='linear')(residual3)

    dnn = Model(inputs=input_layer, outputs=output)

    dnn.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='mse',
        metrics=[rmse]
    )

    return dnn


# MA06 - Advanced Optimizer: Learning Rate Scheduler
def create_model_MA06(input_dimension: int):
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


# MA07 - Hybrid Model (Combining Linear and Nonlinear Components)
def create_model_MA07(input_dimension: int):
    input_layer = Input(shape=(input_dimension,))

    # Linear component
    linear = Dense(1, activation='linear')(input_layer)

    # Nonlinear component
    nonlinear = Dense(128, activation='relu')(input_layer)
    nonlinear = BatchNormalization()(nonlinear)
    nonlinear = Dropout(0.3)(nonlinear)
    nonlinear = Dense(64, activation='relu')(nonlinear)
    nonlinear = BatchNormalization()(nonlinear)
    nonlinear = Dropout(0.3)(nonlinear)
    nonlinear = Dense(32, activation='relu')(nonlinear)
    nonlinear = BatchNormalization()(nonlinear)

    # Merge linear and nonlinear components
    merged = Concatenate()([linear, nonlinear])
    output = Dense(1, activation='linear')(merged)

    dnn = Model(inputs=input_layer, outputs=output)

    dnn.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='mse',
        metrics=[rmse]
    )

    return dnn


# MA08 - Other Optimizers
def create_model_MA08(input_dimension: int):
    dnn = Sequential([
        # Input Layer
        Input(shape=(input_dimension,)),
        Dense(128),
        BatchNormalization(),
        Activation('relu'),
        Dropout(0.5),

        # Hidden Layer 1
        Dense(64),
        BatchNormalization(),
        Activation('relu'),
        Dropout(0.5),

        # Hidden Layer 2
        Dense(32),
        BatchNormalization(),
        Activation('relu'),
        Dropout(0.5),

        # Output Layer
        Dense(1, activation='linear')
    ])

    dnn.compile(optimizer='adam', loss='mse', metrics=[rmse])

    return dnn


# MA09 - Simple and wide model
def create_model_MA09(input_dimension: int):
    dnn = Sequential([
        Input(shape=(input_dimension,)),
        Dense(128, activation='relu', kernel_regularizer=regularizers.l2(1e-4)),
        Dropout(0.4),
        Dense(64, activation='relu', kernel_regularizer=regularizers.l2(1e-4)),
        Dropout(0.3),
        Dense(1, activation='linear')
    ])

    dnn.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='mse',
        metrics=[rmse]
    )

    return dnn


# MA10 - Wide and deep model
def create_model_MA10(input_dimension: int):
    input_layer = Input(shape=(input_dimension,))
    wide = Dense(128, activation='relu')(input_layer)  # Wide component
    deep = Dense(128, activation='relu')(input_layer)
    deep = Dense(64, activation='relu')(deep)
    deep = Dense(32, activation='relu')(deep)

    merged = Concatenate()([wide, deep])
    output_layer = Dense(1, activation='linear')(merged)

    dnn = Model(inputs=input_layer, outputs=output_layer)

    dnn.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='mse',
        metrics=[rmse]
    )

    return dnn


# MA11 - Shallow with Feature Compression
def create_model_MA11(input_dimension: int):
    dnn = Sequential([
        Input(shape=(input_dimension,)),
        Dense(64, activation='relu', kernel_regularizer=regularizers.l2(1e-4)),
        BatchNormalization(),
        Dropout(0.4),
        Dense(32, activation='relu', kernel_regularizer=regularizers.l2(1e-4)),
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


# MA12 - Wider Network (No Gradual Reduction)
def create_model_MA12(input_dimension: int):
    dnn = Sequential([
        Input(shape=(input_dimension,)),
        Dense(128, activation='relu', kernel_regularizer=regularizers.l2(1e-4)),
        BatchNormalization(),
        Dropout(0.4),

        Dense(128, activation='relu', kernel_regularizer=regularizers.l2(1e-4)),
        BatchNormalization(),
        Dropout(0.4),

        Dense(128, activation='relu', kernel_regularizer=regularizers.l2(1e-4)),
        BatchNormalization(),
        Dropout(0.4),

        Dense(1, activation='linear')
    ])

    dnn.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='mse',
        metrics=[rmse]
    )

    return dnn


# MA13 - Regularized Wide & Deep Model
def create_model_MA13(input_dimension: int):
    input_layer = Input(shape=(input_dimension,))

    # Wide part of the model with L2 regularization and Dropout
    wide = Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.01))(input_layer)
    wide = Dropout(0.5)(wide)
    wide = BatchNormalization()(wide)

    # Deep part of the model with L2 regularization, Dropout, and BatchNormalization
    deep = Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.01))(input_layer)
    deep = Dropout(0.5)(deep)
    deep = BatchNormalization()(deep)

    deep = Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.01))(deep)
    deep = Dropout(0.5)(deep)
    deep = BatchNormalization()(deep)

    deep = Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.01))(deep)
    deep = Dropout(0.5)(deep)
    deep = BatchNormalization()(deep)

    # Merge the wide and deep components
    merged = Concatenate()([wide, deep])

    # Output layer
    output_layer = Dense(1, activation='linear')(merged)

    # Create the model
    dnn_model = Model(inputs=input_layer, outputs=output_layer)

    # Compile the model
    dnn_model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='mse',
        metrics=[rmse]
    )

    return dnn_model
