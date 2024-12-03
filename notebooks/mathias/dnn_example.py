
from tensorflow.keras.callbacks import EarlyStopping

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=10,  # Stop after 10 epochs with no improvement
    restore_best_weights=True
)
model = Sequential()

# Input Layer
input_dim=293
lr = 0.02

model.add(Dense(256, activation='relu', input_dim=input_dim))
model.add(BatchNormalization())  # Optional: Normalize activations
model.add(Dropout(0.3))          # Optional: Regularization to prevent overfitting

# Hidden Layers
model.add(Dense(128, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.3))

model.add(Dense(64, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.3))

model.add(Dense(32, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.3))

# Output Layer
model.add(Dense(1, activation='linear'))  # Linear activation for regression

# Compile the model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=lr),  # Optimizer
    loss=rmse,                               # Loss function
    metrics=['mean_squared_error', 'mean_absolute_error',rmse]    # Metrics
)

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=20,  # Stop after 10 epochs with no improvement
    restore_best_weights=True
)

history = model.fit(
    x_train_s,
    y_train,
    validation_data=(x_test_s, y_test),
    epochs=100,
    batch_size=64,
    callbacks=[early_stop],
    verbose=1
)

clear_output( wait=True )
plot_res( history )
get_score( model )