from importdataset import load_cifar10
from visualization import show_images_per_class, class_names
from preprocessing import normalize_images, one_hot_encode_labels
from utils import print_shape_and_dtype

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