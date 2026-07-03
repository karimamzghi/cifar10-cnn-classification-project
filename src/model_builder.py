# Create Reusable CNN model architectures

from pyexpat import model

from tensorflow import keras
from tensorflow.keras import layers

from src import config

def build_dropout_cnn():
    model = keras.Sequential(
        [
            keras.Input(shape=config.INPUT_SHAPE),

            layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
            layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),

            layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
            layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),

            layers.Flatten(),

            layers.Dense(64, activation="relu"),
            layers.Dropout(0.5),

            layers.Dense(config.NUM_CLASSES, activation="softmax"),
        ],
        name="dropout_cnn"
    )

    return model

def build_batchnorm_cnn():

    model = keras.Sequential(
        [
            keras.Input(shape=config.INPUT_SHAPE),

            layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
            layers.BatchNormalization(),
            layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
            layers.BatchNormalization(),
            layers.MaxPooling2D(pool_size=(2, 2)),

            layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
            layers.BatchNormalization(),
            layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
            layers.BatchNormalization(),
            layers.MaxPooling2D(pool_size=(2, 2)),

            layers.Flatten(),

            layers.Dense(64, activation="relu"),
            layers.BatchNormalization(),
            layers.Dropout(0.5),

            layers.Dense(config.NUM_CLASSES, activation="softmax"),
        ],
        name="batchnorm_cnn"
    )

    return model
