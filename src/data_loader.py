import os
import numpy as np
from src import config

#   Load preprocessed CIFAR-10 arrays from Drive (fast).
def load_data():
    x_train = np.load(os.path.join(config.DATA_DIR, "x_train.npy"))
    y_train = np.load(os.path.join(config.DATA_DIR, "y_train.npy"))
    x_test = np.load(os.path.join(config.DATA_DIR, "x_test.npy"))
    y_test = np.load(os.path.join(config.DATA_DIR, "y_test.npy"))

    print(f"x_train: {x_train.shape} | y_train: {y_train.shape}")
    print(f"x_test:  {x_test.shape} | y_test:  {y_test.shape}")

    return x_train, y_train, x_test, y_test
