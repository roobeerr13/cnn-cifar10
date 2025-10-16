# Clasificador de Imágenes CIFAR-10 con CNN y Flask
https://github.com/roobeerr13/cnn-cifar10.git

**Proyecto de Marco Verdú y Roberto Jiménez**

## Descripción

Este proyecto desarrolla e implementa una Red Neuronal Convolucional (CNN) para clasificar imágenes del conocido dataset CIFAR-10. El modelo se entrena desde cero y, una vez guardado, se integra con una aplicación web interactiva construida con Flask.

La aplicación web permite a los usuarios subir sus propias imágenes para clasificarlas en tiempo real y visualiza de forma dinámica el historial de entrenamiento del modelo (precisión y pérdida) a través de gráficos interactivos con una estética de neón.

## Estructura del Proyecto

El repositorio está organizado de la siguiente manera:

- `main.py`: Script principal que se encarga de entrenar el modelo (si no existe un modelo pre-entrenado), evaluar su rendimiento final y lanzar la aplicación web Flask.
- `reporte.md`: Archivo autogenerado que documenta la precisión y pérdida finales del modelo en el conjunto de prueba.
- `modelo_cifar10.h5`: El modelo de Keras pre-entrenado y guardado.
- `history.pkl`: Archivo que almacena las métricas de entrenamiento (precisión y pérdida) por época.

- **`data/`**: Contiene la lógica para cargar el dataset CIFAR-10.
- **`modelo/`**: Define la arquitectura de la red neuronal (`modelo.py`) y la lógica de entrenamiento (`training_model/train_model.py`).
- **`utils/`**: Incluye funciones auxiliares para preprocesamiento de imágenes, visualización y otras utilidades.
- **`flask_web/`**: Contiene todos los archivos relacionados con la interfaz web de Flask.
  - `templates/index.html`: Estructura HTML de la página principal.
  - `static/style.css`: Estilos CSS para la apariencia de la web (tema neón).
  - `static/script.js`: Código JavaScript para la interactividad del cliente (carga de imágenes, predicciones y gráficos).

## Instalación y Uso

Sigue estos pasos para poner en marcha el proyecto:

1.  **Clonar el repositorio:**
    ```bash
    git clone <URL-DEL-REPOSITORIO>
    cd cnn-cifar10
    ```

2.  **Crear un entorno virtual e instalar dependencias:**
    Se recomienda utilizar un entorno virtual para gestionar las dependencias del proyecto.
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # En Windows: .venv\Scripts\activate
    pip install -r requirements.txt
    ```
    *(Nota: Asegúrate de tener un archivo `requirements.txt` con todas las librerías necesarias).* 

3.  **Ejecutar la aplicación:**
    El siguiente comando iniciará el proceso. Si no encuentra un archivo `modelo_cifar10.h5`, comenzará el entrenamiento del modelo. Una vez finalizado (o si el modelo ya existía), lanzará el servidor web de Flask.
    ```bash
    python main.py
    ```

4.  **Acceder a la interfaz web:**
    Abre tu navegador y visita la dirección `http://127.0.0.1:8080` (o el puerto que se indique en la terminal) para interactuar con el clasificador.

## Requisitos

Las principales librerías utilizadas en este proyecto son:

- `tensorflow`
- `flask`
- `numpy`
- `Pillow` (PIL)

Un listado completo se encuentra en el archivo `requirements.txt`.

## Resultados

El rendimiento final del modelo (precisión y pérdida) sobre el conjunto de datos de prueba se calcula automáticamente al ejecutar `main.py` y se documenta en el archivo `reporte.md`.
