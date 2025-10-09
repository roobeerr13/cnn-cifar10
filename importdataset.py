# Importar librerías necesarias
import tensorflow as tf
from tensorflow import keras
from keras.datasets import cifar10
import numpy as np
import matplotlib.pyplot as plt

# Cargar los datos
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Verificar dimensiones y tipos de datos
print("x_train shape:", x_train.shape)
print("y_train shape:", y_train.shape)
print("x_test shape:", x_test.shape)
print("y_test shape:", y_test.shape)
print("Tipo de x_train:", x_train.dtype)
print("Tipo de y_train:", y_train.dtype)