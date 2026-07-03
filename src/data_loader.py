import os
import numpy as np

from src import config
from src.dataset_processing import prepare_data

# Load CIFAR-10 processed data.
def load_data():
    drive_file = os.path.join(config.DRIVE_DATA_DIR, "x_train.npy")
    local_file = os.path.join(config.DATA_DIR, "x_train.npy")

    if os.path.exists(drive_file):
        print("Loading cached data from Google Drive:", config.DRIVE_DATA_DIR)

        return tuple(
            np.load(os.path.join(config.DRIVE_DATA_DIR, f"{name}.npy"))
            for name in config.SPLIT_NAMES
        )

    if os.path.exists(local_file):
        print("Loading cached data from local project:", config.DATA_DIR)

        return tuple(
            np.load(os.path.join(config.DATA_DIR, f"{name}.npy"))
            for name in config.SPLIT_NAMES
        )

    print("Cache not found. Preparing CIFAR-10 data from scratch...")
    return prepare_data()
