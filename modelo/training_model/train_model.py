import os
import pickle
import io
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint

# Import project utilities
from modelo.modelo import crear_modelo


def train_full(x_train, y_train_cat, x_test, y_test_cat,
               input_shape=(32, 32, 3), num_classes=10,
               batch_size=64, epochs=100,
               model_path='modelo_cifar10.h5', history_path='history.pkl', checkpoint_path='modelo_cifar10_best.h5'):
    """
    Entrena un modelo CNN desde cero usando aumentos de datos y callbacks.

    Parámetros:
    - x_train: numpy array (normalizado)
    - y_train_cat: etiquetas one-hot
    - x_test, y_test_cat: datos de validación

    Guarda el mejor modelo en `model_path` y el historial en `history_path`.
    Devuelve el objeto History.
    """

    modelo = crear_modelo(input_shape=input_shape, num_classes=num_classes)
    modelo.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    datagen = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        zoom_range=0.1,
        shear_range=0.05
    )
    datagen.fit(x_train)

    callbacks = [
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1),
        EarlyStopping(monitor='val_accuracy', patience=10, verbose=1, restore_best_weights=True),
        ModelCheckpoint(checkpoint_path, monitor='val_accuracy', save_best_only=True, mode='max', verbose=1)
    ]

    steps_per_epoch = max(1, x_train.shape[0] // batch_size)

    history = modelo.fit(
        datagen.flow(x_train, y_train_cat, batch_size=batch_size),
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        validation_data=(x_test, y_test_cat),
        callbacks=callbacks,
        verbose=1
    )

    # Si existe el checkpoint, cargar sus pesos
    if os.path.exists(checkpoint_path):
        modelo.load_weights(checkpoint_path)

    # Guardar el modelo final y el historial
    modelo.save(model_path)
    with open(history_path, 'wb') as f:
        pickle.dump(history.history, f)

    return history


def plot_history_to_buffer(history):
    """Genera una imagen PNG en memoria con loss y accuracy desde history (dict o History)."""
    if hasattr(history, 'history'):
        h = history.history
    else:
        h = history

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    if 'loss' in h and 'val_loss' in h:
        ax[0].plot(h['loss'], label='train_loss')
        ax[0].plot(h['val_loss'], label='val_loss')
        ax[0].set_title('Pérdida')
        ax[0].set_xlabel('Épocas')
        ax[0].set_ylabel('Loss')
        ax[0].legend()

    if 'accuracy' in h and 'val_accuracy' in h:
        ax[1].plot(h['accuracy'], label='train_acc')
        ax[1].plot(h['val_accuracy'], label='val_acc')
        ax[1].set_title('Precisión')
        ax[1].set_xlabel('Épocas')
        ax[1].set_ylabel('Accuracy')
        ax[1].legend()

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    return buf
