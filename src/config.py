# Dataset configuration
NUM_CLASSES = 10
INPUT_SHAPE = (32, 32, 3)

# Split configuration (70 / 15 / 15)
TEST_SIZE = 0.15
VAL_SIZE = 0.15

# Class names
CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]

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

# Model configuration
MODEL_NAME = "model_1_baseline_cnn"

# Paths
DATA_DIR = "./data/processed"
MODELS_DIR = "./models"
RESULTS_PATH = "./results/model_tracking.csv"
CONFUSION_DIR = "./results/confusion_matrices"
PLOTS_DIR = "./results/plots"

