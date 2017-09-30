# todo: numpy vs pytorch

import numpy as np

from nn_utils import sigmoid, sigmoid_prime, relu, relu_prime

def initialize_parameters( layer_dims):  #[12288, 20, 7, 5, 1]
    np.random.seed(1)
    
    params = {}

    L = len(layer_dims)
    for l in range(1, L):
        params['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) / np.sqrt(layer_dims[l-1]) #*0.01
        params['b' + str(l)] = np.zeros((layer_dims[l], 1))

    return params


def forward_propagation( X, params):
    #[12288, 20, 7, 5, 1]
    
    L = len(params) // 2   # L=4
    
    caches = []   # (A_prev, Z)

    A_prev = X
    for l in range(1, L): #1..3
        Z = np.dot( params["W"+str(l)], A_prev) + params["b"+str(l)]
        A = relu( Z)
        
        caches.append( (A_prev ,Z))
        A_prev = A

    Z = np.dot( params["W"+str(L)], A_prev) + params["b"+str(L)]
    AL = sigmoid( Z)
    
    caches.append( (A_prev, Z) )
    
    return AL, caches


def compute_cost( A, Y):
    m = Y.shape[1]
    
    cost = -(1.0/m) * np.sum( np.multiply(Y, np.log(A)) + np.multiply( (1-Y), np.log(1-A)))
    
    return cost


def compute_grads( dA, W, cache, activation = "relu"):
    A_prev, Z = cache
    
    if "relu" == activation:
        dZ = dA * relu_prime( Z)

    if "sigmoid" == activation:
        dZ = dA * sigmoid_prime( Z)
    
    dW = np.dot( dZ, A_prev.T)
    db = dZ.sum( axis = 1, keepdims = True)

    dA_prev = np.dot(W.T, dZ)
        
    return dA_prev, dW, db


    
def backward_propagation( AL, Y, params, caches):
    #[12288, 20, 7,  5,  1]
    #      [z1, z2, z3, z4]
    
    grads = {}
    
    L = len(params) // 2
    inv_m = 1.0 / Y.shape[1]
        
    activation = "sigmoid"
    dA = -np.divide(Y, AL) + np.divide(1-Y, 1-AL)
    for l in reversed( range( 1, L+1)):  # 4,..,1
        dA_prev, dW, db = compute_grads( dA, params["W"+str(l)], caches[l-1], activation)
        grads["dW"+str(l)] = inv_m * dW
        grads["db"+str(l)] = inv_m * db
        dA = dA_prev
        activation = "relu"

    return dA_prev, grads


def update_parameters( grads, params, learning_rate):
    L = len(params) // 2
    
    for l in reversed( range( 1, L+1)): #4,..,1
        params["W"+str(l)] = params["W"+str(l)] - learning_rate * grads["dW"+str(l)]
        params["b"+str(l)] = params["b"+str(l)] - learning_rate * grads["db"+str(l)]
        
    return params


def nn_model( X, Y, layers_dim, epochs = 2500, learning_rate = 0.0075):
    np.random.seed(1)
    
    parameters = initialize_parameters( layers_dim)

    costs = []
    for epoch in range(epochs):
        AL, caches = forward_propagation( X, parameters)
           
        cost = compute_cost( AL, Y)

        _, grads = backward_propagation( AL, Y, parameters, caches)

        parameters = update_parameters( grads, parameters, learning_rate)

        if 0 == (epoch%100):
            costs.append( cost)
            print("Epoch: {} - Cost: {}".format(epoch, cost))
    
        
    return parameters, costs


def nn_predict( X, params):
    m = X.shape[1]
    
    probs = np.zeros((1,m))
    
    y_hat, _ = forward_propagation( X, params)
    
    probs[ y_hat > 0.5] = 1
    
    return probs
    
    
