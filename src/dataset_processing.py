import os
import numpy as np

from keras.datasets import cifar10
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

from src import config

# Download CIFAR-10, create a 70/15/15 stratified split, normalize images, one-hot encode labels, and cache arrays.
def prepare_data():
    os.makedirs(config.DATA_DIR, exist_ok=True)

    if os.path.exists("/content/drive"):
        os.makedirs(config.DRIVE_DATA_DIR, exist_ok=True)

    (x_train_original, y_train_original), (x_test_original, y_test_original) = cifar10.load_data()

    x_all = np.concatenate([x_train_original, x_test_original], axis=0)
    y_all = np.concatenate([y_train_original, y_test_original], axis=0).reshape(-1)

    x_temp, x_test, y_temp, y_test = train_test_split(
        x_all,
        y_all,
        test_size=config.TEST_SIZE,
        stratify=y_all,
        random_state=config.STATE,
    )

    val_fraction = config.VAL_SIZE / (1 - config.TEST_SIZE)

    x_train, x_val, y_train, y_val = train_test_split(
        x_temp,
        y_temp,
        test_size=val_fraction,
        stratify=y_temp,
        random_state=config.STATE,
    )

    x_train = x_train.astype("float32") / 255.0
    x_val = x_val.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    y_train = to_categorical(y_train, num_classes=config.NUM_CLASSES)
    y_val = to_categorical(y_val, num_classes=config.NUM_CLASSES)
    y_test = to_categorical(y_test, num_classes=config.NUM_CLASSES)

    arrays = {
        "x_train": x_train,
        "y_train": y_train,
        "x_val": x_val,
        "y_val": y_val,
        "x_test": x_test,
        "y_test": y_test,
    }

    for name, array in arrays.items():
        np.save(os.path.join(config.DATA_DIR, f"{name}.npy"), array)

        if os.path.exists("/content/drive"):
            np.save(os.path.join(config.DRIVE_DATA_DIR, f"{name}.npy"), array)

    print("Data prepared and cached.")
    print("Train:", x_train.shape, y_train.shape)
    print("Validation:", x_val.shape, y_val.shape)
    print("Test:", x_test.shape, y_test.shape)

    return tuple(arrays[name] for name in config.SPLIT_NAMES)
