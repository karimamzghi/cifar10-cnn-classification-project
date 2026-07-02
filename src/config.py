# Dataset configuration
NUM_CLASSES = 10
INPUT_SHAPE = (32, 32, 3)

# Compiling configuration
ADAM_OP = "adam"
CATEGORICAL_CROSSENTROPY = "categorical_crossentropy"
ACCURACY = "accuracy"

# Training configuration
BATCH_SIZE = 64
EPOCHS = 20
LEARNING_RATE = 0.001

# Model configuration
MODEL_NAME = "model_1_baseline_cnn"

# Paths
RESULTS_PATH = "./results/model_tracking.csv"
MODELS_DIR = "./models"
