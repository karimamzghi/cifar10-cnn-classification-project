# Create Reusable CNN model architectures

from pyexpat import model

from tensorflow import keras
from tensorflow.keras import layers

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

from src import config

# Build Model 4: CNN with Dropout: Add a Dropout layer after the Dense layer to reduce overfitting.
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

# Build Model 5: CNN with Batch Normalization: Add Batch Normalization layers after each convolutional layer to improve training stability and performance.
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


# Build Model 6: CNN with Data Augmentation: From model 5, add data augmentation at the beginning of the model. Keep Batch Normalization and Dropout.
def build_augmented_cnn():
    data_augmentation = keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.1),
            layers.RandomZoom(0.1),
        ],
        name="data_augmentation"
    )

    model = keras.Sequential(
        [
            keras.Input(shape=config.INPUT_SHAPE),

            data_augmentation,

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
        name="augmented_cnn"
    )

    return model

# Build Model 8: Larger CNN with Data Augmentation: Increase the number of convolutional layers and filters, and add data augmentation at the beginning of the model. Keep Batch Normalization and Dropout.
def build_larger_augmented_cnn():
    data_augmentation = keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.1),
            layers.RandomZoom(0.1),
        ],
        name="data_augmentation"
    )

    model = keras.Sequential(
        [
            keras.Input(shape=config.INPUT_SHAPE),

            data_augmentation,

            layers.Conv2D(32, (3, 3), activation="relu"),
            layers.BatchNormalization(),
            layers.Conv2D(32, (3, 3), activation="relu"),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),

            layers.Conv2D(64, (3, 3), activation="relu"),
            layers.BatchNormalization(),
            layers.Conv2D(64, (3, 3), activation="relu"),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),

            layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
            layers.BatchNormalization(),
            layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),

            layers.Flatten(),

            layers.Dense(128, activation="relu"),
            layers.BatchNormalization(),
            layers.Dropout(0.5),

            layers.Dense(config.NUM_CLASSES, activation="softmax"),
        ],
        name="larger_augmented_cnn"
    )

    return model

# Build Model 9: MobileNetV2 with Frozen Base: Use the MobileNetV2 architecture with pre-trained weights from ImageNet, freeze the base layers, and add custom classification layers on top.
def build_mobilenet_frozen():
    base_model = MobileNetV2(
        weights="imagenet",
        include_top=False,
        input_shape=config.INPUT_SHAPE
    )

    base_model.trainable = False

    model = keras.Sequential(
        [
            keras.Input(shape=config.INPUT_SHAPE),

            layers.Lambda(lambda x: preprocess_input(x * 255.0)),

            base_model,

            layers.GlobalAveragePooling2D(),

            layers.Dense(128, activation="relu"),
            layers.Dropout(0.5),

            layers.Dense(config.NUM_CLASSES, activation="softmax"),
        ],
        name="mobilenet_frozen"
    )

    return model


# Build Model 10: MobileNetV2 with Fine-Tuning: Use the MobileNetV2 architecture with pre-trained weights from ImageNet, unfreeze the top layers of the base model for fine-tuning, and add custom classification layers on top.
def build_mobilenet_finetuned():
    base_model = MobileNetV2(
        weights="imagenet",
        include_top=False,
        input_shape=config.INPUT_SHAPE
    )

    base_model.trainable = True

    # Freeze most layers, fine-tune only the top part
    for layer in base_model.layers[:-30]:
        layer.trainable = False

    model = keras.Sequential(
        [
            keras.Input(shape=config.INPUT_SHAPE),

            layers.Lambda(lambda x: preprocess_input(x * 255.0)),

            base_model,

            layers.GlobalAveragePooling2D(),

            layers.Dense(128, activation="relu"),
            layers.Dropout(0.5),

            layers.Dense(config.NUM_CLASSES, activation="softmax"),
        ],
        name="mobilenet_finetuned"
    )

    return model
