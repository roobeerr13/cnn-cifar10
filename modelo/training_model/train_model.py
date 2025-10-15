import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
import io
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical

# === Importar tus funciones existentes ===
from data.import_dataset import load_cifar10
from utils.preprocessing import normalize_images, one_hot_encode_labels
from utils.visualization import class_names
from modelo.modelo import crear_modelo

# === Variables globales ===
MODEL_PATH = "modelos_guardados/cifar10_model.h5"
modelo = None
history = None

# === Función de entrenamiento ===
def train_model(epochs=5):
    global modelo, history

    x_train, y_train, x_test, y_test = load_cifar10()
    x_train = normalize_images(x_train)
    x_test = normalize_images(x_test)
    y_train_cat = one_hot_encode_labels(y_train)
    y_test_cat = one_hot_encode_labels(y_test)

    modelo = crear_modelo(input_shape=(32, 32, 3), num_classes=10)
    modelo.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

    datagen = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True
    )
    datagen.fit(x_train)

    history = modelo.fit(
        datagen.flow(x_train, y_train_cat, batch_size=64),
        epochs=epochs,
        validation_data=(x_test, y_test_cat),
        verbose=0
    )

    modelo.save(MODEL_PATH)

    # Crear las gráficas en memoria
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(history.history["loss"], label="Entrenamiento")
    ax[0].plot(history.history["val_loss"], label="Validación")
    ax[0].set_title("Pérdida")
    ax[0].set_xlabel("Épocas")
    ax[0].set_ylabel("Loss")
    ax[0].legend()

    ax[1].plot(history.history["accuracy"], label="Entrenamiento")
    ax[1].plot(history.history["val_accuracy"], label="Validación")
    ax[1].set_title("Precisión")
    ax[1].set_xlabel("Épocas")
    ax[1].set_ylabel("Accuracy")
    ax[1].legend()

    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)

    return "✅ Entrenamiento completado y modelo guardado.", buf
