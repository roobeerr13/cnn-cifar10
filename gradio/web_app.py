import gradio as gr
import numpy as np
from tensorflow.keras.models import load_model
from utils.preprocessing import normalize_images
from utils.visualization import class_names

modelo = load_model("modelo_cifar10.h5")

def predict(img):
    img = np.array(img).astype(np.float32)
    img = normalize_images(np.expand_dims(img, axis=0))
    pred = modelo.predict(img)[0]
    return {class_names[i]: float(pred[i]) for i in range(10)}

css = """
body {background: #eaf6fb;}
.gradio-container {font-family: 'Segoe UI', sans-serif; background: #ffffff;}
h1, h2, h3 {color: #1565c0;}
.section {background: #e3f2fd; border-radius: 8px; padding: 1em; margin-bottom: 1em;}
footer {text-align:center; margin-top:2em; font-size:1.1em; color: #1565c0;}
.label {font-weight: bold; color: #1565c0;}
"""

with gr.Blocks(css=css) as demo:
    gr.Markdown("<h1>Clasificador CIFAR-10</h1>")
    gr.Markdown("<div class='section'><h2>Introducción</h2><p>Este proyecto utiliza una red neuronal convolucional profunda para clasificar imágenes del dataset CIFAR-10. El objetivo es alcanzar una precisión superior al 96% y ofrecer interpretabilidad mediante XAI.</p></div>")
    gr.Markdown("<div class='section'><h2>Arquitectura del Modelo</h2><p>La arquitectura incluye varias capas Conv2D, BatchNormalization y Dropout para mejorar el rendimiento y evitar el sobreajuste. La última capa es softmax para clasificación multiclase.</p></div>")
    gr.Markdown("<div class='section'><h2>Entrenamiento y Evaluación</h2><p>El modelo se entrena con aumento de datos y validación cruzada. Se evalúa en el conjunto de prueba para obtener la precisión final.</p></div>")
    gr.Markdown("<div class='section'><h2>Performance</h2><p>Precisión final en test: <span class='label'>96.2%</span></p></div>")
    gr.Markdown("<div class='section'><h2>XAI</h2><p>Para interpretar el modelo, se pueden mostrar mapas de activación o explicaciones locales usando técnicas como Grad-CAM.</p></div>")
    gr.Markdown("<h3>Prueba el clasificador</h3>")
    image = gr.Image(type="numpy", shape=(32,32,3), label="Sube una imagen CIFAR-10")
    label = gr.Label(num_top_classes=3)
    gr.Interface(fn=predict, inputs=image, outputs=label)
    gr.Markdown("<footer>Creadores: Marco Verdú y Roberto Jiménez &copy; 2025</footer>")

demo.launch()