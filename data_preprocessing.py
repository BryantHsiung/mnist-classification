import numpy as np
from keras.datasets import mnist

def load_mnist():
    """Loads, reshapes, and normalizes the MNIST dataset."""
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = np.reshape(x_train, (len(x_train), 28*28)) / 255.0
    x_test = np.reshape(x_test, (len(x_test), 28*28)) / 255.0
    return (x_train, y_train), (x_test, y_test)
