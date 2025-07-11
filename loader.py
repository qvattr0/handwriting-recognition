import pickle
import gzip
from tensorflow.keras.datasets import mnist # type: ignore

def load_data():
    """Load, process, and output training data from the MNIST dataset on handwritten digits.

    Returns:
        tuple: A tuple containing:
            - training_data (list[tuple[numpy.ndarray, int]]): A list of 60,000 (image, label) tuples representing training data. Each image is a 784-element array.
            - testing_data (list[tuple[numpy.ndarray, int]]): A list of 10,000 (image, label) tuples representing test data. Each image is a 784-element array.
    """
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    # normalizing the data
    X_train = X_train.astype("float32")/255
    X_test = X_test.astype("float32")/255

    # reshaping the images as vectors
    X_train = X_train.reshape(60000, 784, 1)
    X_test = X_test.reshape(10000,784, 1)

    # reconfiguring the labels to be one-hot vectors
    # conformance to the output layer of the network
    # i.e.: "3" is [0 0 0 1 0 0 0 0 0 0]
    y_train_onehot = []
   
    for y in y_train:
        blank = [0] * 10
        blank[y] = 1
        y_train_onehot.append(blank)

    y_train = y_train_onehot
    
    training_data = list(zip(X_train, y_train))
    testing_data  = list(zip(X_test, y_test))
    return training_data, testing_data