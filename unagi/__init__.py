import numpy as np

import unagi.optim as optim
import unagi.utils as utils


class nn(object):
	def __init__( self, layers_dim, framework = None): #None: numpy, tf, torch
		self._initializer = "xavier"
		self._layers_dim = layers_dim
		self.optimizer = optim.GradientDescent()


	def set_seed( self, seed):
		np.random.seed(1)


	def initialize_parameters(self, layers_dim):
	    params = {}

	    sqrt2 = np.sqrt(2)

	    L = len(layers_dim)
	    for l in range(1, L):
	        if "he" == self._initializer:
	            scale = sqrt2 / np.sqrt(layers_dim[l-1])
	        elif "xavier" == self._initializer:
	            scale = 1.0 / np.sqrt(layers_dim[l-1])
	        else:
	            scale = 0.01

	        params['W' + str(l)] = np.random.randn(layers_dim[l], layers_dim[l-1]) * scale
	        params['b' + str(l)] = np.zeros((layers_dim[l], 1))

	    return params


	def forward_propagation( self, X, params, drop_out = 0):
	    L = len(params) // 2   # Example: L=4, [n_x, 20, 7, 5, 1]
	    
	    caches = []   # (A_prev, DropoutMask, Z)

	    keep_prob = 1 - drop_out
	    
	    A_prev = X
	    for l in range(1, L): #1..3
	        Z = np.dot( params["W"+str(l)], A_prev) + params["b"+str(l)]
	        A = utils.relu( Z)
	        
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
	    AL = utils.sigmoid( Z)
	    caches.append( (A_prev, -1, Z) )
	    
	    return AL, caches


	def compute_grads( self, inv_m, dA, W, cache, lambd_factor = 0, activation = "relu"):
	    A_prev, D, Z = cache
	    
	    if "relu" == activation:
	        dZ = dA * utils.relu_prime( Z)

	    if "sigmoid" == activation:
	        dZ = dA * utils.sigmoid_prime( Z)
	    
	    dW = inv_m * np.dot( dZ, A_prev.T)    
	    if 0 != lambd_factor:
	        dW = dW + lambd_factor * W     
	    db = inv_m * dZ.sum( axis = 1, keepdims = True)

	    dA_prev = np.dot(W.T, dZ)
	    
	    return dA_prev, dW, db


	def backward_propagation( self, AL, Y, params, caches, lambd_factor, drop_out):
	    grads = {}
	    
	    L = len(params) // 2
	    
	    inv_m = 1. / Y.shape[1]

	    activation = "sigmoid"
	    dA = -np.divide(Y, AL) + np.divide(1-Y, 1-AL)
	    for l in reversed( range( 1, L+1)):
	        dA_prev, grads["dW"+str(l)], grads["db"+str(l)] = \
	            self.compute_grads( inv_m, dA, params["W"+str(l)], caches[l-1], lambd_factor, activation)
	            
	        if 0 < drop_out and 1 < l:
	            _, D, _ = caches[l-2]
	            dA_prev = np.multiply( dA_prev, D)
	            dA_prev = dA_prev / (1-drop_out)
	            
	        dA = dA_prev
	        activation = "relu"

	    return grads, dA_prev


	def compute_cost( self, logits, labels, params, lambd_factor = 0):
	    m = labels.shape[1]

	    l2_reg_cost = 0
	    if 0 != lambd_factor: # lambd_factor = lambd / m
	        L = len(params) // 2
	        for l in range(1,L+1):
	            l2_reg_cost += np.sum(np.square(params["W"+str(l)]))
	        l2_reg_cost = (lambd_factor*0.5) * l2_reg_cost

	    logprobs = np.multiply(-np.log(logits),labels) + np.multiply(-np.log(1 - logits), 1 - labels)
	    cost = 1./m * np.sum(logprobs) + l2_reg_cost

	    return cost


	def compute_batch( self, X, Y, params, lambd_factor, drop_out):
		AL, caches = self.forward_propagation( X, params, drop_out)
		cost = self.compute_cost( AL, Y, params, lambd_factor)
		grads, _ = self.backward_propagation( AL, Y, params, caches, lambd_factor, drop_out)
		
		params = self.optimizer.step( params, grads)

		return params, cost

		
	def train( self, X, Y, epochs = 2500, batch_size = 32, lambd = 0, drop_out = 0):
	    self.parameters = self.initialize_parameters( self._layers_dim)

	    lambd_factor = 0

	    self.costs = []
	    self.optimizer.zero_grad()

	    for epoch in range(epochs):
	        epoch_cost = 0
	        
	        minibatches = utils.random_minibatch( X, Y, batch_size)
	        num_minibatches = len(minibatches)

	        for batch in minibatches:
	            (batch_X, batch_Y) = batch
	            
	            if 0 != lambd:
	                lambd_factor = lambd / batch_X.shape[1]
	        
	            self.parameters, cost = self.compute_batch( 
					batch_X, batch_Y,
					self.parameters,
					lambd_factor,
					drop_out)

	            epoch_cost += cost / num_minibatches

	        if 0 == epoch%100:
	            print("Epoch: {}. Cost: {}".format(epoch, epoch_cost))
	            
	        if 0 == epoch%5:
	            self.costs.append( epoch_cost)


	def predict( self, X):
		m = X.shape[1]

		probs = np.zeros((1,m))

		# noTE: Never perform drop out during prediction
		y_hat, _ = self.forward_propagation( X, self.parameters, drop_out = 0)

		probs[ y_hat > 0.5] = 1

		return probs
