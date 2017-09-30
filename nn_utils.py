import numpy as np
import h5py

def sigmoid(Z):
    return 1.0 / (1 + np.exp(-Z))

def sigmoid_prime(Z):
    return sigmoid(Z) * (1 - sigmoid(Z))

def relu(Z):
    return np.maximum(0, Z)

def relu_prime(Z):
    return (Z >= 0) * 1


def load_data( train_filename, test_filename ):
    train_dataset = h5py.File( train_filename, "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:]) # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:]) # your train set labels

    test_dataset = h5py.File( test_filename, "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:]) # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:]) # your test set labels

    classes = np.array(test_dataset["list_classes"][:]) # the list of classes
    
    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))
    
    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes
