import gradio as gr
import numpy as np
from tensorflow.keras.models import load_model
from preprocessing import normalize_images
from visualization import class_names

modelo = load_model("modelo_cifar10.h5")  # Guarda tu modelo tras entrenar

def predict(img):
    img = np.array(img).astype(np.float32)
    img = normalize_images(np.expand_dims(img, axis=0))
    pred = modelo.predict(img)[0]
    return {class_names[i]: float(pred[i]) for i in range(10)}

css = """
body {background: #f0f4f8;}
.gradio-container {font-family: 'Segoe UI', sans-serif;}
h1 {color: #2c3e50;}
footer {text-align:center; margin-top:2em; font-size:1.1em;}
"""

with gr.Blocks(css=css) as demo:
    gr.Markdown("# Clasificador CIFAR-10")
    gr.Markdown("Creadores: **Marco Verdú** y **Roberto Jiménez**")
    image = gr.Image(type="numpy", shape=(32,32,3), label="Sube una imagen CIFAR-10")
    label = gr.Label(num_top_classes=3)
    gr.Interface(fn=predict, inputs=image, outputs=label)
    gr.Markdown("<footer>Proyecto CNN CIFAR-10 &copy; 2025</footer>")

demo.launch()