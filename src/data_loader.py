# load_data allows us to access the cached dataset splits (train/val/test) as numpy arrays.
# If the cache is missing, it calls dataset_preprocessing.prepare_data()
# load_data returns (x_train, y_train, x_val, y_val, x_test, y_test)

import os

import numpy as np

from src import config
from src.dataset_preprocessing import SPLIT_NAMES, prepare_data


def load_data():
    first_file = os.path.join(config.DATA_DIR, "x_train.npy")

    if not os.path.exists(first_file):
        print("Cache not found, running dataset_preprocessing and loading the data...")
        return prepare_data()

    return tuple(
        np.load(os.path.join(config.DATA_DIR, f"{name}.npy"))
        for name in SPLIT_NAMES
    )
