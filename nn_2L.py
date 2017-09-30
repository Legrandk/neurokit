import numpy as np

import numpy as np

from nn_utils import sigmoid, sigmoid_prime, relu, relu_prime

"""
def sigmoid(z):
    return 1.0 / (1 + np.exp(-z))


def ReLU(z):
    return np.maximum(0, z)

def ReLU_deriv(z):
    return (z > 0) * z
"""

def initialize_parameters( n_x, n_h, n_y):
    W1 = np.random.randn( n_h, n_x) * 0.01
    b1 = np.zeros((n_h, 1))
    W2 = np.random.randn( n_y, n_h) * 0.01
    b2 = np.zeros((n_y, n_y))
    
    params = {
            "W1": W1, "b1": b1,
            "W2": W2, "b2": b2 }
    
    return params
    
  
def forward_propagation( X, params):
    W1 = params["W1"]
    b1 = params["b1"]
    W2 = params["W2"]
    b2 = params["b2"]
        
    Z1 = np.dot( W1, X) + b1
    #A1 = np.tanh(Z1)
    #A1 = sigmoid(Z1)
    A1 = relu(Z1)
    
    Z2 = np.dot( W2, A1) + b2
    A2 = sigmoid(Z2)
    
    cache = {
            "Z1": Z1, "A1": A1,
            "Z2": Z2, "A2": A2 }
    
    return A2, cache
    

def compute_cost( A, Y):
    m = Y.shape[1]
    
    logprobs = np.multiply( Y, np.log(A)) + np.multiply( ( 1-Y), np.log(1-A))
    
    cost = -(1.0/m) * logprobs.sum()
    
    cost = np.squeeze(cost) # turns [[17]] into 17 
    
    return cost


def backward_propagation( X, Y, params, cache):
    A2 = cache["A2"]
    A1 = cache["A1"]
    Z1 = cache["Z1"]
    
    W2 = params["W2"]
    
    inv_m = 1.0 / X.shape[1]
    
    
    dZ2 = A2 - Y # (1,m)
    dW2 = inv_m * np.dot( dZ2, A1.T)  #(n_y, n_h)
    db2 = inv_m * dZ2.sum( axis=1, keepdims=True)
    #dZ1 = np.dot( W2.T, dZ2) * (1 - np.power(A1,2))   #tanh'
    #dZ1 = np.dot( W2.T, dZ2) * (A1 * (1-A1))          #Sigmoid'
    dZ1 = np.dot( W2.T, dZ2) * relu_prime(Z1)          #Relu'
    
    dW1 = inv_m * np.dot(dZ1, X.T)
    db1 = inv_m * dZ1.sum( axis = 1, keepdims=True)
    
    grads = {
        "dW2": dW2,
        "db2": db2,
        "dW1": dW1,
        "db1": db1 }
    
    return grads


def update_params( params, grads, learning_rate = 1.2):
    params["W2"] = params["W2"] - learning_rate * grads["dW2"]
    params["b2"] = params["b2"] - learning_rate * grads["db2"]

    params["W1"] = params["W1"] - learning_rate * grads["dW1"]
    params["b1"] = params["b1"] - learning_rate * grads["db1"]
    
    return params


def predict( X, params):
    y_pred, _ = forward_propagation( X, params)
    
    return ( y_pred > 0.5)*1


def nn_model( X_train, Y_train, hidden_layer = 4, epochs = 10000 , learning_rate = 1.2):
    n_x = X_train.shape[0]   # num of neurons input layer
    n_h = hidden_layer       # num of neurons hidden layer
    n_y = Y_train.shape[0]   # num of neurons ouput layer
    
    params = initialize_parameters( n_x, n_h, n_y)
    
    
    for epoch in range(epochs):
        
        A2, cache = forward_propagation( X_train, params)
        
        cost = compute_cost( A2, Y_train)
        
        grads = backward_propagation( X_train, Y_train, params, cache)
        
        params = update_params( params, grads, learning_rate)
        
        
        if 0 == epoch%1000:
            print("Epoch: {} - Cost: {}".format(epoch, cost))
        
    return params

