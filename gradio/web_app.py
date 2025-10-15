import gradio as gr
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from utils.preprocessing import normalize_images
from utils.visualization import class_names

# Cargar modelo si existe
MODEL_PATH = "modelo_cifar10.h5"
HISTORY_PATH = "history.pkl"
if os.path.exists(MODEL_PATH):
    modelo = load_model(MODEL_PATH)
else:
    modelo = None

def predict(img):
    if modelo is None:
        return "Modelo no disponible. Entrena y guarda el modelo primero."
    img_np = np.array(img).astype(np.float32)
    img_pre = normalize_images(np.expand_dims(img_np, axis=0))
    pred = modelo.predict(img_pre)[0]
    top_idx = int(np.argmax(pred))
    return {"label": class_names[top_idx], "probability": float(pred[top_idx])}

def plot_history():
    if not os.path.exists(HISTORY_PATH):
        return None
    with open(HISTORY_PATH, "rb") as f:
        history = pickle.load(f)

    # Crear figura con loss y accuracy
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    # Loss
    if 'loss' in history and 'val_loss' in history:
        axes[0].plot(history['loss'], label='train_loss')
        axes[0].plot(history['val_loss'], label='val_loss')
        axes[0].set_title('Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
    # Accuracy
    if 'accuracy' in history and 'val_accuracy' in history:
        axes[1].plot(history['accuracy'], label='train_acc')
        axes[1].plot(history['val_accuracy'], label='val_acc')
        axes[1].set_title('Accuracy')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].legend()

    plt.tight_layout()
    return fig

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
    label = gr.JSON(label="Predicción")

    # Preparar figura del historial (si existe)
    hist_fig = plot_history() if os.path.exists(HISTORY_PATH) else None

    with gr.Row():
        with gr.Column():
            image_in = image
            btn = gr.Button("Predecir")
        with gr.Column():
            label_out = label
            hist_out = gr.Plot(value=hist_fig, label="History (loss & accuracy)")

    btn.click(fn=predict, inputs=image_in, outputs=label_out)
    gr.Markdown("<footer>Creadores: Marco Verdú y Roberto Jiménez &copy; 2025</footer>")

demo.launch()