# views.py
import numpy as np
import matplotlib.pyplot as plt

# Class names for CIFAR-10
class_names = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]

def show_images_per_class(x, y, class_names):
    """
    Display one image for each CIFAR-10 class with its label.
    """
    plt.figure(figsize=(12, 6))
    for i in range(10):
        idx = np.where(y == i)[0][0]
        plt.subplot(2, 5, i+1)
        plt.imshow(x[idx])
        plt.title(class_names[i])
        plt.axis('off')
    plt.tight_layout()
    plt.show()