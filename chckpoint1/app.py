import gradio as gr


def predict(image):
    """
    Prediction placeholder.

    The trained ResNet-18 model and Grad-CAM
    will be connected in a later milestone.
    """

    return {
        "Normal": 0.0,
        "Pneumonia": 0.0,
        "COVID-19": 0.0
    }


demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(),
    title="Chest X-Ray Classification",
    description=(
        "Educational prototype for three-class "
        "chest X-ray classification."
    )
)


if __name__ == "__main__":
    demo.launch()
