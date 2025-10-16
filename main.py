from data.import_dataset import load_cifar10
from utils.visualization import class_names
from utils.preprocessing import normalize_images, one_hot_encode_labels
from utils.utils import print_shape_and_dtype
from modelo.modelo import crear_modelo
from modelo.training_model.train_model import train_full
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from flask import Flask, render_template, request, jsonify
import numpy as np
from PIL import Image
import io
import pickle
from tensorflow.keras.models import load_model

# Initialize Flask app
app = Flask(__name__, template_folder='flask_web/templates', static_folder='flask_web/static')

# Load dataset
x_train, y_train, x_test, y_test = load_cifar10()

# Print shapes and dtypes
print_shape_and_dtype("x_train", x_train)
print_shape_and_dtype("y_train", y_train)
print_shape_and_dtype("x_test", x_test)
print_shape_and_dtype("y_test", y_test)

# Normalize images
x_train = normalize_images(x_train)
x_test = normalize_images(x_test)

# One-hot encode labels
y_train_cat = one_hot_encode_labels(y_train)
y_test_cat = one_hot_encode_labels(y_test)

# Confirm image shape
print("Image shape:", x_train[0].shape)

# Create and compile model
modelo = crear_modelo(input_shape=(32, 32, 3), num_classes=10)
modelo.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
modelo.summary()

# Data augmentation
datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    zoom_range=0.1,
    shear_range=0.05
)
datagen.fit(x_train)

# Callbacks
checkpoint_path = "modelo_cifar10_best.h5"
callbacks = [
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1),
    EarlyStopping(monitor='val_accuracy', patience=10, verbose=1, restore_best_weights=True),
    ModelCheckpoint(checkpoint_path, monitor='val_accuracy', save_best_only=True, mode='max', verbose=1)
]

MODEL_PATH = "modelo_cifar10.h5"

# Train or load model
if not os.path.exists(MODEL_PATH):
    print("No pre-trained model found. Starting training...")
    history = train_full(x_train, y_train_cat, x_test, y_test_cat,
                         input_shape=(32, 32, 3), num_classes=10,
                         batch_size=64, epochs=100,
                         model_path=MODEL_PATH, history_path='history.pkl', checkpoint_path='modelo_cifar10_best.h5')
else:
    print(f"Model found at {MODEL_PATH}. Loading model.")
    modelo = load_model(MODEL_PATH)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    if file:
        try:
            img = Image.open(io.BytesIO(file.read())).convert('RGB')
            img = img.resize((32, 32))
            img_np = np.array(img).astype(np.float32)
            
            img_pre = normalize_images(np.expand_dims(img_np, axis=0))
            
            preds = modelo.predict(img_pre)[0]
            
            result = {class_names[i]: float(preds[i]) for i in range(len(class_names))}
            
            return jsonify(result)
        except Exception as e:
            return jsonify({'error': str(e)})

@app.route('/history')
def history():
    with open('history.pkl', 'rb') as f:
        history_data = pickle.load(f)
    return jsonify(history_data)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port)
