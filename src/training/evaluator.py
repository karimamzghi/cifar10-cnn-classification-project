import os
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_score,
    recall_score
)
from src import config


def evaluate_model(model, x_test, y_test):
    # Sanity check: confirm we're using the new 70/15/15 split (15% of 60,000 = 9,000) and not the original Keras default test set (10,000)
    print("x_test shape:", x_test.shape)
    print("Expected: ~9000 (15% of 60000)")

    # Evaluate the baseline model on the held-out test set 
    test_loss, test_accuracy = model.evaluate(
        x_test,
        y_test,
        verbose=0
    )
    print("\nTest loss:", test_loss)
    print("Test accuracy:", test_accuracy)

    return test_loss, test_accuracy


def get_predictions(model, x_test, y_test):    
    y_pred_probs = model.predict(x_test)

    test_predictions = np.argmax(y_pred_probs, axis=1)
    test_labels = np.argmax(y_test, axis=1)


    return test_labels, test_predictions, y_pred_probs

def calculate_precision_recall(test_labels, test_predictions):
    precision = precision_score(
        test_labels,
        test_predictions,
        average="weighted",
        zero_division=0
    )

    recall = recall_score(
        test_labels,
        test_predictions,
        average="weighted",
        zero_division=0
    )
    print(f"Precision: {precision:.4f}  Recall: {recall:.4f}")

    return precision, recall


# Draw the confusion matrix for the test set predictions
def save_confusion_matrix(test_labels, test_predictions, output_path, title):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    cm = confusion_matrix(test_labels, test_predictions)

    ax = plt.subplots(figsize=(10, 10))

    # Create a ConfusionMatrixDisplay object and plot the confusion matrix
    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=config.CLASS_NAMES
    )

    # Plot the confusion matrix
    display.plot(
        ax=ax,
        values_format="d",
        xticks_rotation=45
    )

    plt.title(plt.title)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')

    # Save the confusion matrix plot to the specified output path
    plt.savefig(output_path, bbox_inches="tight")
    plt.show()

    return output_path
