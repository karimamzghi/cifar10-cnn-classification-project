# Dataset configuration
NUM_CLASSES = 10
INPUT_SHAPE = (32, 32, 3)
TARGET_SIZE = (32, 32)  # For image resizing during prediction

# Split configuration (70 / 15 / 15)
TEST_SIZE = 0.15
VAL_SIZE = 0.15

# Class names
CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]

# SplitNames configuration
SPLIT_NAMES = ["x_train", "y_train", "x_val", "y_val", "x_test", "y_test"]

# Compiling configuration
ADAM_OP = "adam"
CATEGORICAL_CROSSENTROPY = "categorical_crossentropy"
ACCURACY = "accuracy"

# config.py
STATE = 42

# Training configuration
BATCH_SIZE = 64
EPOCHS = 20
LEARNING_RATE = 0.001

# Early stopping configuration
EARLY_STOPPING_PATIENCE = 5
EARLY_STOPPING_MONITOR = "val_loss"

# Paths
DATA_DIR = "./data/processed"
MODELS_DIR = "./models"
RESULTS_PATH = "./results/model_tracking.csv"
CONFUSION_DIR = "./results/confusion_matrices"
PLOTS_DIR = "./results/plots"
