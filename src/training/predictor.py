"""Load a saved model and predict a single uploaded image (for deployment)."""

import numpy as np
from tensorflow.keras.models import load_model
from src import config


def load_trained_model(model_path):
    model = load_model(model_path)
    return model


def preprocess_single_image(image, target_size):
    image = image.resize(target_size)

   # Ensure image always has 3 channels
    image = image.convert("RGB")

    # Resize image
    image = image.resize(target_size)

    # Convert to NumPy array
    image_array = np.array(image)

    # Normalize pixels
    image_array = image_array.astype("float32") / 255.0

    # Add batch dimension
    image_array = np.expand_dims(image_array, axis=0)

    return image_array


def predict_image(model, image):
    processed_image = preprocess_single_image(
        image,
        target_size=config.TARGET_SIZE
    )

  # Predict probabilities
    prediction_probs = model.predict(
        processed_image,
        verbose=0
    )[0]

    # Best prediction
    predicted_index = np.argmax(prediction_probs)

    predicted_class = config.CLASS_NAMES[predicted_index]

    confidence = float(
        prediction_probs[predicted_index] * 100
    )

    # Probability for every class
    probabilities = {
        config.CLASS_NAMES[i]: float(prediction_probs[i])
        for i in range(len(config.CLASS_NAMES))
    }

    # Sort probabilities (highest first)
    probabilities = dict(
        sorted(
            probabilities.items(),
            key=lambda item: item[1],
            reverse=True
        )
    )

    # Top-3 predictions
    top_indices = np.argsort(prediction_probs)[::-1][:3]

    top_predictions = [
        {
            "class": config.CLASS_NAMES[i],
            "confidence": round(
                float(prediction_probs[i] * 100),
                2
            )
        }
        for i in top_indices
    ]

    return {
        "predicted_class": predicted_class,
        "confidence": round(confidence, 2),
        "probabilities": probabilities,
        "top_predictions": top_predictions
    }
