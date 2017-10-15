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


def forward_propagation( X, params, drop_out = 0):
    L = len(params) // 2   # Example: L=4, [n_x, 20, 7, 5, 1]
    
    caches = []   # (A_prev, DropoutMask, Z)

    keep_prob = 1 - drop_out
    
    A_prev = X
    for l in range(1, L): #1..3
        Z = np.dot( params["W"+str(l)], A_prev) + params["b"+str(l)]
        A = relu( Z)
        
        D = -1
        if 1 > keep_prob:
            D = np.random.rand( A.shape[0], A.shape[1]) < keep_prob

            # Bulletproof: Don't want a D full of zeros vector   
            if True == np.any(~D.any(axis=0)):
                D[np.random.randint(D.shape[0]),~D.any(axis=0)] = 1
            
            A = np.multiply( A, D)
            A = A / keep_prob  #inverted step
        
        caches.append( (A_prev, D, Z))
        A_prev = A

    Z = np.dot( params["W"+str(L)], A_prev) + params["b"+str(L)]
    AL = sigmoid( Z)
    caches.append( (A_prev, -1, Z) )
    
    return AL, caches


def compute_cost( A, Y, params, lambd_factor = 0):
    m = Y.shape[1]
    
    l2_reg_cost = 0
    if 0 != lambd_factor: # lambd_factor = lambd / m
        L = len(params) // 2
        for l in range(1,L+1):
            l2_reg_cost += np.sum(np.square(params["W"+str(l)]))
        l2_reg_cost = (lambd_factor*0.5) * l2_reg_cost

    logprobs = np.multiply(-np.log(A),Y) + np.multiply(-np.log(1 - A), 1 - Y)
    cost = 1./m * np.sum(logprobs) + l2_reg_cost

    return cost


def compute_grads( inv_m, dA, W, cache, lambd_factor = 0, activation = "relu"):
    
    A_prev, D, Z = cache
    
    if "relu" == activation:
        dZ = dA * relu_prime( Z)

    if "sigmoid" == activation:
        dZ = dA * sigmoid_prime( Z)
    
    dW = inv_m * np.dot( dZ, A_prev.T)    
    if 0 != lambd_factor:
        dW = dW + lambd_factor * W     
    db = inv_m * dZ.sum( axis = 1, keepdims = True)

    dA_prev = np.dot(W.T, dZ)
    
    return dA_prev, dW, db


    
def backward_propagation( AL, Y, params, caches, lambd_factor = 0, drop_out = 0):
    grads = {}
    
    L = len(params) // 2
    
    inv_m = 1. / Y.shape[1]

    activation = "sigmoid"
    dA = -np.divide(Y, AL) + np.divide(1-Y, 1-AL)
    for l in reversed( range( 1, L+1)):
        dA_prev, grads["dW"+str(l)], grads["db"+str(l)] = \
            compute_grads( inv_m, dA, params["W"+str(l)], caches[l-1], lambd_factor, activation)
            
        if 0 < drop_out and 1 < l:
            _, D, _ = caches[l-2]
            dA_prev = np.multiply( dA_prev, D)
            dA_prev = dA_prev / (1-drop_out)
            
        dA = dA_prev
        activation = "relu"

    return grads, dA_prev

    
def update_parameters( grads, params, learning_rate, lambd_factor = 0):
    L = len(params) // 2
    
    for l in reversed( range( 1, L+1)):
        dW = grads["dW"+str(l)]
        if 0 != lambd_factor:
            dW = dW + lambd_factor * params["W"+str(l)]
        
        params["W"+str(l)] = params["W"+str(l)] - learning_rate * dW
        params["b"+str(l)] = params["b"+str(l)] - learning_rate * grads["db"+str(l)]
        
    return params


def gradient_check(X, Y,  grads, params, epsilon = 1e-7):
    pass


def nn_model( X, Y, layers_dim, epochs = 2500, initialization = "he", learning_rate = 0.0075, lambd = 0, drop_out = 0):
    parameters = initialize_parameters( layers_dim, initialization)

    lambd_factor = 0
    if 0 != lambd:
        lambd_factor = lambd / X.shape[1]

    costs = []
    for epoch in range(epochs):
        AL, caches = forward_propagation( X, parameters, drop_out)
           
        cost = compute_cost( AL, Y, parameters, lambd_factor)

        grads, _ = backward_propagation( AL, Y, parameters, caches, lambd_factor, drop_out)
        
        parameters = update_parameters( grads, parameters, learning_rate)

        if 0 == (epoch%100):
            costs.append( cost)
            print("Epoch: {} - Cost: {}".format(epoch, cost))
    
        
    return parameters, costs


def nn_predict( X, params):
    m = X.shape[1]
    
    probs = np.zeros((1,m))
    
    # noTE: Never perform dropout during prediction
    y_hat, _ = forward_propagation( X, params, drop_out = 0)
    
    probs[ y_hat > 0.5] = 1
    
    return probs
