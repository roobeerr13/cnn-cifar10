import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
import io
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from modelo.training_model.train_model import train_full, plot_history_to_buffer
from data.import_dataset import load_cifar10
from utils.preprocessing import normalize_images, one_hot_encode_labels

# === Importar tus funciones existentes ===
from data.import_dataset import load_cifar10
from utils.preprocessing import normalize_images, one_hot_encode_labels
from utils.visualization import class_names
from modelo.modelo import crear_modelo

# === Variables globales ===
MODEL_PATH = "modelo_cifar10.h5"
modelo = None
history = None


# === Función para clasificar imágenes ===
def predict_image(img):
    global modelo

    if modelo is None:
        try:
            modelo = load_model(MODEL_PATH)
        except Exception:
            return "⚠️ Entrena el modelo primero.", None

    img = img.resize((32, 32))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    preds = modelo.predict(img_array)
    preds = preds[0]

    # Convertir a dict: clase -> probabilidad
    result = {class_names[i]: float(preds[i]) for i in range(len(class_names))}

    return result


def train_button_handler(epochs):
    """Handler que carga datos, preprocesa y llama a train_full con el número de épocas."""
    try:
        x_train, y_train, x_test, y_test = load_cifar10()
        x_train = normalize_images(x_train)
        x_test = normalize_images(x_test)
        y_train_cat = one_hot_encode_labels(y_train)
        y_test_cat = one_hot_encode_labels(y_test)

        history = train_full(x_train, y_train_cat, x_test, y_test_cat, epochs=int(epochs), model_path=MODEL_PATH)
        buf = plot_history_to_buffer(history)
        return "✅ Entrenamiento completado y modelo guardado.", buf
    except Exception as e:
        return f"❌ Error durante el entrenamiento: {e}", None

# === Interfaz Gradio ===
def run_app():
    with gr.Blocks(title="Clasificador CIFAR-10 con Gradio") as demo:
        gr.Markdown("## 🧠 Clasificador CIFAR-10")
        gr.Markdown("Entrena el modelo, visualiza las métricas y prueba tus imágenes.")

        with gr.Tab("1️⃣ Entrenar modelo"):
            epochs = gr.Slider(1, 30, value=10, step=1, label="Número de épocas")
            train_btn = gr.Button("🚀 Entrenar modelo")
            output_msg = gr.Textbox(label="Estado")
            output_plot = gr.Image(label="Gráficas de entrenamiento")

            train_btn.click(train_button_handler, inputs=epochs, outputs=[output_msg, output_plot])

        with gr.Tab("2️⃣ Clasificar imágenes"):
            gr.Markdown("Sube una imagen y el modelo te dirá qué clase es y su porcentaje de precisión.")
            image_input = gr.Image(type="pil", label="Sube una imagen")
            output_label = gr.Label(num_top_classes=3, label="Predicción (Top 3)")
            predict_btn = gr.Button("🔍 Clasificar imagen")

            predict_btn.click(predict_image, inputs=image_input, outputs=output_label)

    demo.launch()


if __name__ == '__main__':
    run_app()
