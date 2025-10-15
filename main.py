from data.import_dataset import load_cifar10
from utils.visualization import show_images_per_class, class_names
from utils.preprocessing import normalize_images, one_hot_encode_labels
from utils.utils import print_shape_and_dtype
from modelo.modelo import crear_modelo
from modelo.training_model.train_model import train
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pickle
# Load dataset
x_train, y_train, x_test, y_test = load_cifar10()

# Print shapes and dtypes
print_shape_and_dtype("x_train", x_train)
print_shape_and_dtype("y_train", y_train)
print_shape_and_dtype("x_test", x_test)
print_shape_and_dtype("y_test", y_test)

# Visualize images
show_images_per_class(x_train, y_train, class_names)

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
    horizontal_flip=True
)
datagen.fit(x_train)

# Entrenamiento con aumento de datos
history = modelo.fit(
    datagen.flow(x_train, y_train_cat, batch_size=64),
    epochs=50,
    validation_data=(x_test, y_test_cat)
)

modelo.save("modelo_cifar10.h5")
with open("history.pkl", "wb") as f:
    pickle.dump(history.history, f)

print("Modelo guardado en 'modelo_cifar10.h5' y historial en 'history.pkl'.")