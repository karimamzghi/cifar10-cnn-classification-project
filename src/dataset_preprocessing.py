# This functin download CIFAR-10 DataSet, 
# split 70/15/15 and stratified,
# Applies normalizatin and one-hot encode
# It caches the data to disk as .npy files.
# this function is called once, and then the data is loaded from disk in ~2s.

import os

import numpy as np
from keras.datasets import cifar10
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from src.config import SPLIT_NAMES

from src import config


def prepare_data():
    os.makedirs(config.DATA_DIR, exist_ok=True)

    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    # Verify and check the data by calling shape of the data
    print("x_train: ", x_train.shape)
    print("y_train: ", y_train.shape)

    print("x_test: ", x_test.shape)
    print("y_test: ", y_test.shape)

# Combine full dataset — Keras' default 50k/10k split isn't 70/15/15
    x_all = np.concatenate([x_train, x_test], axis=0)
    y_all = np.concatenate([y_train, y_test], axis=0).reshape(-1)

    # split 15% for test
    x_temp, x_test, y_temp, y_test = train_test_split(
        x_all, y_all,
        test_size=config.TEST_SIZE,
        stratify=y_all,
        random_state=config.STATE,
    )
    # From the remaining 85%, take  15% of the total for the validation set
    val_fraction = config.VAL_SIZE / (1 - config.TEST_SIZE)
    x_train, x_val, y_train, y_val = train_test_split(
        x_temp, y_temp,
        test_size=val_fraction,
        stratify=y_temp,
        random_state=config.STATE,
    )

    # Normalising the data by dividing every pixel by 255.
    # Image pixels from will displayed with values range from 0-1 instead of  0-255    
    x_train = x_train.astype("float32") / 255.0
    x_val = x_val.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    print("Train Data after normalisation: ", x_train[:10])
    print("Validation Data after normalisation: ", x_val[:10])
    print("Test Data after normalisation: ", x_test[:10])

    print("x_train min:", x_train.min())
    print("x_train max:", x_train.max())
    print("x_train dtype:", x_train.dtype)

    #Feature Engineering: Hardcoding labels using one-hot encoding
    # Perform one-hot encoding for 10 classes
    y_train = to_categorical(y_train, num_classes=config.NUM_CLASSES)
    y_val = to_categorical(y_val, num_classes=config.NUM_CLASSES)
    y_test = to_categorical(y_test, num_classes=config.NUM_CLASSES)

    print("Train Data after encoding: ", y_train[:10])
    print("\nValidation Data after encoding: ", y_val[:10])
    print("\nTest Data after encoding: ", y_test[:10])

    # Cache the data to disk as .npy files for faster loading in future runs
    arrays = dict(zip(SPLIT_NAMES,
                      [x_train, y_train, x_val, y_val, x_test, y_test]))
    for name, arr in arrays.items():
        np.save(os.path.join(config.DATA_DIR, f"{name}.npy"), arr)

    print(f"Cached {len(x_train)} train / {len(x_val)} val / "
          f"{len(x_test)} test samples to {config.DATA_DIR}")
    return tuple(arrays[name] for name in SPLIT_NAMES)
