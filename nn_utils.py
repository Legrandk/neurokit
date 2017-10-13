import numpy as np
import h5py

def sigmoid(Z):
    return 1.0 / (1 + np.exp(-Z))

def sigmoid_prime(Z):
    return sigmoid(Z) * (1 - sigmoid(Z))

def relu(Z):
    return np.maximum(0, Z)

def relu_prime(Z):
    return (Z >= 0)


def params_to_vector( params):
    shapes = []
    
    vector = -1
    
    L = len(params) // 2
    for i in range(1, L+1):
        w_vector = np.reshape( params["W"+str(i)], (-1,1))
        b_vector = np.reshape( params["b"+str(i)], (-1,1))
        
        if False == isinstance(vector, int):
            vector = np.concatenate( (vector, w_vector, b_vector))
        else:
            vector = np.concatenate( (w_vector, b_vector))
            
        shapes.append( params["W"+str(i)].shape)
        shapes.append( params["b"+str(i)].shape)    
    
    return vector, shapes

def grads_to_vector( grads):
    shapes = []
    
    vector = -1
    
    L = len(grads) // 2
    for i in range(1, L+1):
        dw_vector = np.reshape( grads["dW"+str(i)], (-1,1))
        db_vector = np.reshape( grads["db"+str(i)], (-1,1))
        
        if False == isinstance(vector, int):
            vector = np.concatenate( (vector, dw_vector, db_vector))
        else:
            vector = np.concatenate( (dw_vector, db_vector))
            
        shapes.append( grads["dW"+str(i)].shape)
        shapes.append( grads["db"+str(i)].shape)    
    
    return vector, shapes


def vector_to_parameters( vector, shapes):
    params = {}
    
    row = 0
    cont = 1
    for i in range(0, len(shapes), 2):
        shape = shapes[i]
        row_limit = row+shape[0]*shape[1]        
        params["W"+str(cont)] = np.reshape( vector[row:row_limit, 0], (shape[0], shape[1]))
        row = row_limit

        shape = shapes[i+1]
        row_limit = row+shape[0]*shape[1]        
        params["b"+str(cont)] = np.reshape( vector[row:row_limit, 0], (shape[0], shape[1]))
        row = row_limit
        cont = cont + 1
    
    return params


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

