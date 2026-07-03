import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

import gradio as gr
from src.training.predictor import load_trained_model, predict_image

MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "model_1_baseline_cnn.keras")

model = load_trained_model(MODEL_PATH)


def classify_image(image):
    result = predict_image(model, image)

    # Full breakdown as a markdown table, top 3 bolded
    top3_classes = {p["class"] for p in result["top_predictions"]}

    rows = ["| Class | Probability |", "|---|---|"]
    for class_name, prob in result["probabilities"].items():
        pct = f"{prob * 100:.2f}%"
        if class_name in top3_classes:
            rows.append(f"| **{class_name}** ⭐ | **{pct}** |")
        else:
            rows.append(f"| {class_name} | {pct} |")

    full_table = "\n".join(rows)

    return result["probabilities"], full_table


demo = gr.Interface(
    fn=classify_image,
    inputs=gr.Image(type="pil"),
    outputs=[
        gr.Label(num_top_classes=3, label="Top 3 Predictions"),
        gr.Markdown(label="All Class Probabilities"),
    ],
    title= "CIFAR-10 CNN Image Classifier",
    description="Upload an image and the baseline CNN model will predict one of the CIFAR-10 classes. "
                "The top 3 predictions (⭐) are highlighted above; the full probability breakdown is shown below."
)

demo.launch(share=True)
