# cnn-cifar10

# Clasificador CIFAR-10 con CNN

Proyecto de Marco Verdú y Roberto Jiménez

## Descripción

Este proyecto entrena una red neuronal convolucional profunda para clasificar imágenes del dataset CIFAR-10, alcanzando una precisión superior al 96%. Incluye una interfaz web interactiva con Gradio y un diseño personalizado.

## Estructura

- `main.py`: Flujo principal de entrenamiento y evaluación.
- `modelo.py`: Arquitectura de la CNN.
- `train_model.py`: Función para entrenar el modelo.
- `web_app.py`: Interfaz web con Gradio.
- `preprocessing.py`, `visualization.py`, `utils.py`: Funciones auxiliares.

## Uso

1. **Entrenamiento**
   ```bash
   python main.py
   ```
   El modelo se guarda como `modelo_cifar10.h5`.

2. **Interfaz web**
   ```bash
   python web_app.py
   ```
   Accede a la web local para probar el clasificador.

## Requisitos

- Python 3.8+
- TensorFlow
- Gradio
- Matplotlib
- Numpy

Instala dependencias:
```bash
pip install tensorflow gradio matplotlib numpy
```

## Créditos

Creadores: **Marco Verdú** y **Roberto Jiménez**

---