# Normalization and one-hot encoding

import numpy as np
from tensorflow.keras.utils import to_categorical

def normalize_images(x):
    """
    Normalize image pixel values to the range [0, 1].
    """
    x_norm = x.astype('float32') / 255.0
    print("Normalized range:", x_norm.min(), "to", x_norm.max())
    return x_norm

def one_hot_encode_labels(y, num_classes=10):
    """
    Convert class labels to one-hot encoded format.
    """
    y_cat = to_categorical(y, num_classes)
    print("One-hot shape:", y_cat.shape)
    return y_cat