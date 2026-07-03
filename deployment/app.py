import gradio as gr
from src.training.predictor import load_trained_model, predict_image

MODEL_PATH = "models/model_1_baseline_cnn.keras"

model = load_trained_model(MODEL_PATH)


def classify_image(image):
    result = predict_image(model, image)

    return result["probabilities"]


demo = gr.Interface(
    fn=classify_image,
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(num_top_classes=3),
    title="CIFAR-10 CNN Image Classifier",
    description="Upload an image and the baseline CNN model will predict one of the CIFAR-10 classes."
)

demo.launch(share=True)
