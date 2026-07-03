"""Compile and train models. Knows nothing about metrics or plots."""

import os
import time
from xml.parsers.expat import model
from keras.callbacks import EarlyStopping
from src import config


def compile_model(model, optimizer, loss, metrics):
    if metrics is None:
        metrics = ["accuracy"]

    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=metrics
    )

    return model


def train_model(data, model, batch_size, epochs, early_stopping=False):
    x_train, y_train, x_val, y_val, x_test, y_test = data

    callbacks = []
    if early_stopping:
        callbacks.append(EarlyStopping(monitor="val_loss", patience=5,
                                       restore_best_weights=True))

    start = time.time()
   
    # Train the model using the training data and validate it using the validation data
    history = model.fit(
        x_train,
        y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(x_val, y_val),
        callbacks=callbacks,
        verbose=1
    )

    return history, round(time.time() - start, 1)

#   Save a trained Keras model.
def save_model(model, model_name, models_dir):
    os.makedirs(models_dir, exist_ok=True)

    model_path = os.path.join(
        models_dir,
        f"{model_name}.keras"
    )

    model.save(model_path)

    print(f"Model saved to: {model_path}")

    return model_path
