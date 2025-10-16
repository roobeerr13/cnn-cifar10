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
from data.import_dataset import load_cifar10
import pandas as pd

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
    # Tema oscuro con acentos neón
    css = """
    body { background: #0b0f1a; color: #e6f2ff; }
    .header { background: linear-gradient(90deg,#0f1724,#001529); color: #00e5ff; padding: 12px; border-radius: 8px; box-shadow: 0 6px 20px rgba(0,0,0,0.6); }
    .neon { color: #00e5ff; text-shadow: 0 0 8px rgba(0,229,255,0.6); }
    .card { background: rgba(255,255,255,0.03); border: 1px solid rgba(0,229,255,0.06); padding: 12px; border-radius: 8px; }
    .btn { background: linear-gradient(90deg,#00e5ff,#6a00ff); color: black; font-weight: 600; }
    """

    with gr.Blocks(css=css, title="Clasificador CIFAR-10 - Dark Neon") as demo:
        # Header simple, sin la sección de bienvenida inútil
        with gr.Row():
            gr.Markdown("<div class='header'><h2 class='neon'>Clasificador CIFAR-10</h2></div>")

        with gr.Tabs():
            # Pestaña Clasificar
            with gr.TabItem("Clasificar"):
                with gr.Row():
                    with gr.Column(scale=2):
                        gr.Markdown("### Subir imagen para clasificar")
                        image_input = gr.File(label="Sube una imagen (archivo)")
                        predict_btn = gr.Button("🔍 Clasificar imagen", elem_id="predict_btn")
                        # Mostrar resultados: label y tabla
                        output_label = gr.Label(num_top_classes=3, label="Predicción (Top 3)")
                        prob_table = gr.Dataframe(headers=["Clase","Probabilidad (%)"], label="Probabilidades (Top 3)")
                    with gr.Column(scale=1):
                        gr.Markdown("### Vista previa")
                        preview = gr.Image(label="Preview", interactive=False)

                def preview_and_predict(file):
                    if file is None:
                        return None, {}, []
                    # abrir archivo con PIL
                    try:
                        img = Image.open(file.name)
                    except Exception:
                        return None, {"error":"No se pudo leer la imagen"}, []
                    preds = predict_image(file)
                    # si predict_image devolvió error dict
                    if isinstance(preds, dict) and 'error' in preds:
                        return img, preds, []

                    # construir tabla top-3
                    items = sorted(preds.items(), key=lambda x: x[1], reverse=True)[:3]
                    table = [[c, round(p*100,2)] for c, p in items]
                    # label component acepta dict
                    return img, preds, table

                predict_btn.click(fn=preview_and_predict, inputs=image_input, outputs=[preview, output_label, prob_table])

            # Pestaña Gráficas
            with gr.TabItem("Gráficas"):
                gr.Markdown("### Métricas de entrenamiento")
                history_fig = load_history_fig('history.pkl')
                if history_fig is not None:
                    gr.Plot(value=history_fig)
                else:
                    gr.Markdown("No se encontró `history.pkl`. Entrena el modelo para ver las gráficas.")

            # Pestaña Dataset con muestras
            with gr.TabItem("Dataset"):
                gr.Markdown("### Muestras del dataset CIFAR-10")
                try:
                    x_train, y_train, x_test, y_test = load_cifar10()
                    # seleccionar una muestra por clase (o primeras 10)
                    imgs = []
                    captions = []
                    for i in range(10):
                        idxs = np.where(y_train == i)[0]
                        if len(idxs) == 0:
                            continue
                        img_np = x_train[idxs[0]]
                        # asegurar rango 0-255 y tipo uint8 para mostrar
                        arr = (img_np * 255).astype('uint8') if img_np.max() <= 1.0 else img_np.astype('uint8')
                        imgs.append(arr)
                        captions.append(class_names[i])
                    gr.Gallery(value=imgs, label="Ejemplos por clase").style(grid=5)
                except Exception as e:
                    gr.Markdown(f"No se pudieron cargar las muestras del dataset: {e}")

    demo.launch()


if __name__ == '__main__':
    run_app()
