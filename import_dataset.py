import numpy as np
from tensorflow.keras.datasets import cifar10

def load_cifar10():
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    print("x_train shape:", x_train.shape)
    print("y_train shape:", y_train.shape)
    print("x_test shape:", x_test.shape)
    print("y_test shape:", y_test.shape)
    print("Tipo de x_train:", x_train.dtype)
    print("Tipo de y_train:", y_train.dtype)
    return x_train, y_train, x_test, y_test
