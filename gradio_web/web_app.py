import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
import io
import os
from tensorflow.keras.models import load_model
from utils.preprocessing import normalize_images
from utils.visualization import class_names
from modelo.modelo import crear_modelo
from PIL import Image

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

    # img puede ser un path (cuando usamos gr.File) o un PIL.Image
    pil_img = None
    try:
        # Si es un objeto con atributo 'name' (gr.File), abrirlo
        if hasattr(img, 'name'):
            pil_img = Image.open(img.name)
        # Si es una ruta
        elif isinstance(img, str):
            pil_img = Image.open(img)
        else:
            pil_img = img
    except Exception:
        return {"error": "No se pudo leer la imagen subida."}

    # Asegurar formato RGB y tamaño 32x32 como en entrenamiento
    try:
        pil_img = pil_img.convert('RGB')
    except Exception:
        pass
    pil_img = pil_img.resize((32, 32))
    img_np = np.array(pil_img).astype(np.float32)

    # Log básico para comprobar recepción correcta
    try:
        print("[predict_image] received image - shape:", img_np.shape, "dtype:", img_np.dtype,
              "min:", img_np.min(), "max:", img_np.max())
    except Exception:
        pass

    img_pre = normalize_images(np.expand_dims(img_np, axis=0))
    # normalize_images ya imprime el rango normalizado

    preds = modelo.predict(img_pre)
    preds = preds[0]

    # Devolver dict con probabilidades
    result = {class_names[i]: float(preds[i]) for i in range(len(class_names))}
    return result


def load_history_fig(history_path='history.pkl'):
    if not os.path.exists(history_path):
        return None
    import pickle
    with open(history_path, 'rb') as f:
        h = pickle.load(f)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    if 'loss' in h and 'val_loss' in h:
        axes[0].plot(h['loss'], label='train_loss')
        axes[0].plot(h['val_loss'], label='val_loss')
        axes[0].set_title('Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
    if 'accuracy' in h and 'val_accuracy' in h:
        axes[1].plot(h['accuracy'], label='train_acc')
        axes[1].plot(h['val_accuracy'], label='val_acc')
        axes[1].set_title('Accuracy')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].legend()
    plt.tight_layout()
    return fig


# (Se eliminó la funcionalidad de entrenamiento desde la UI)

# === Interfaz Gradio ===
def run_app():
    with gr.Blocks(title="Modelo con CIFAR-10 con Gradio") as demo:
        gr.Markdown("## 🧠 Modelo CIFAR-10")
        gr.Markdown("Prueba tus imágenes y visualiza los resultados.")

        # Solo pestaña de clasificación (entrenamiento eliminado)
        with gr.Tab("Clasificar imágenes"):
            gr.Markdown("Sube una imagen y el modelo te dirá qué clase es y su porcentaje de precisión.")
            # usar gr.File para forzar subir archivo y eliminar webcam
            image_input = gr.File(label="Sube una imagen (archivo)")
            predict_btn = gr.Button("🔍 Clasificar imagen")
            output_label = gr.Label(num_top_classes=3, label="Predicción (Top 3)")

            # Gráfica del historial (si existe) justo debajo del upload
            history_plot = load_history_fig('history.pkl')
            plot_component = gr.Plot(value=history_plot, label='Training history (loss & accuracy)')

            predict_btn.click(predict_image, inputs=image_input, outputs=output_label)

    demo.launch()


if __name__ == '__main__':
    run_app()
