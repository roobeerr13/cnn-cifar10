from data.import_dataset import load_cifar10
from utils.visualization import class_names
from utils.preprocessing import normalize_images, one_hot_encode_labels
from utils.utils import print_shape_and_dtype
from modelo.modelo import crear_modelo
from modelo.training_model.train_model import train_full
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from gradio.web_app import predict_image
import numpy as np
import pickle
import os
# Load dataset
x_train, y_train, x_test, y_test = load_cifar10()

# Print shapes and dtypes
print_shape_and_dtype("x_train", x_train)
print_shape_and_dtype("y_train", y_train)
print_shape_and_dtype("x_test", x_test)
print_shape_and_dtype("y_test", y_test)

# (Removed) Visualization of training images to prevent opening 'figure1' on execution

# Normalize images
x_train = normalize_images(x_train)
x_test = normalize_images(x_test)

# One-hot encode labels
y_train_cat = one_hot_encode_labels(y_train)
y_test_cat = one_hot_encode_labels(y_test)

# Confirm image shape
print("Image shape:", x_train[0].shape)
print("In CNNs, shape is (32, 32, 3). In MLPs, images are flattened with .reshape(32*32*3).")

# Crear y mostrar el modelo
modelo = crear_modelo(input_shape=(32, 32, 3), num_classes=10)
modelo.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
modelo.summary()

# Pre train
datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    zoom_range=0.1,
    shear_range=0.05
)
datagen.fit(x_train)
# Ajustes para mejorar convergencia y generalización
batch_size = 64
epochs = 10

# Callbacks: reducir LR, early stopping y guardar el mejor modelo
checkpoint_path = "modelo_cifar10_best.h5"
callbacks = [
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1),
    EarlyStopping(monitor='val_accuracy', patience=10, verbose=1, restore_best_weights=True),
    ModelCheckpoint(checkpoint_path, monitor='val_accuracy', save_best_only=True, mode='max', verbose=1)
]

MODEL_PATH = "modelo_cifar10.h5"

# Si no existe el modelo ya entrenado, entrenar y guardarlo; si existe, cargarlo
if not os.path.exists(MODEL_PATH):
    print("No se encontró modelo preentrenado. Iniciando entrenamiento...")
    history = train_full(x_train, y_train_cat, x_test, y_test_cat,
                         input_shape=(32, 32, 3), num_classes=10,
                         batch_size=64, epochs=100,
                         model_path=MODEL_PATH, history_path='history.pkl', checkpoint_path='modelo_cifar10_best.h5')
else:
    print(f"Modelo encontrado en {MODEL_PATH}. No se reentrena.")

# Lanzar la interfaz web de Gradio
try:
    from gradio.web_app import run_app
except Exception:
    # fallback si el paquete es ejecutado directamente
    from gradio import web_app as web_app_module
    run_app = getattr(web_app_module, 'run_app', None)

if run_app is not None:
    run_app()
else:
    print("No se pudo arrancar la interfaz de Gradio (run_app no encontrado).")
