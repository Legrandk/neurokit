# toDO: numpy vs pytorch

import numpy as np

from nn_utils import sigmoid, sigmoid_prime, relu, relu_prime


def initialize_parameters( layer_dims, method = "rand"):
    params = {}

    sqrt2 = np.sqrt(2)

    L = len(layer_dims)
    for l in range(1, L):
        if "he" == method:
            scale = sqrt2 / np.sqrt(layer_dims[l-1])
        elif "xavier" == method:
            scale = 1.0 / np.sqrt(layer_dims[l-1])
        else:
            scale = 0.01

        params['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) * scale
        params['b' + str(l)] = np.zeros((layer_dims[l], 1))

    return params


def forward_propagation( X, params):
    L = len(params) // 2   # Example: L=4, [n_x, 20, 7, 5, 1]
    
    caches = []   # (A_prev, Z)

    A_prev = X
    for l in range(1, L): #1..3
        Z = np.dot( params["W"+str(l)], A_prev) + params["b"+str(l)]
        A = relu( Z)
        caches.append( (A_prev, Z))
        A_prev = A

    Z = np.dot( params["W"+str(L)], A_prev) + params["b"+str(L)]
    AL = sigmoid( Z)
    caches.append( (A_prev, Z) )
    
    return AL, caches


def compute_cost( A, Y, parameters, lambd = 0):
    m = Y.shape[1]
    
    l2_reg_cost = 0
    
    #if 0 != lambd:
    #    l2_reg_cost = toDO
    
    logprobs = np.multiply(-np.log(A),Y) + np.multiply(-np.log(1 - A), 1 - Y)
    cost = 1./m * np.sum(logprobs) + l2_reg_cost

    return cost


def compute_grads( inv_m, dA, W, cache, activation = "relu"):
    A_prev, Z = cache
    
    if "relu" == activation:
        dZ = dA * relu_prime( Z)

    if "sigmoid" == activation:
        dZ = dA * sigmoid_prime( Z)
    
    dW = inv_m * np.dot( dZ, A_prev.T)
    db = inv_m * dZ.sum( axis = 1, keepdims = True)

    dA_prev = np.dot(W.T, dZ)
        
    return dA_prev, dW, db


    
def backward_propagation( AL, Y, params, caches, lambd = 0):
    grads = {}
    
    L = len(params) // 2
    
    inv_m = 1. / Y.shape[1]

    activation = "sigmoid"
    dA = -np.divide(Y, AL) + np.divide(1-Y, 1-AL)
    for l in reversed( range( 1, L+1)):
        dA_prev, grads["dW"+str(l)], grads["db"+str(l)] = \
            compute_grads( inv_m, dA, params["W"+str(l)], caches[l-1], activation)
        dA = dA_prev
        activation = "relu"

    return grads, dA_prev

    
def update_parameters( grads, params, learning_rate):
    L = len(params) // 2
    
    for l in reversed( range( 1, L+1)):
        params["W"+str(l)] = params["W"+str(l)] - learning_rate * grads["dW"+str(l)]
        params["b"+str(l)] = params["b"+str(l)] - learning_rate * grads["db"+str(l)]
        
    return params


def gradient_check(X, Y,  grads, params, epsilon = 1e-7):
    pass


def nn_model( X, Y, layers_dim, epochs = 2500, initialization = "he", learning_rate = 0.0075, lambd = 0):
    parameters = initialize_parameters( layers_dim, initialization)

    costs = []
    for epoch in range(epochs):
        AL, caches = forward_propagation( X, parameters)
           
        cost = compute_cost( AL, Y, parameters, lambd)

        grads, _ = backward_propagation( AL, Y, parameters, caches, lambd)
        
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
