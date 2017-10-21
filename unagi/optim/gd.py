class GradientDescent:
    def __init__(self, learning_rate = 0.01):
        self._learning_rate = learning_rate


    def zero_grad(self):
        pass


    def set_learning_rate(self, learning_rate):
        self._learning_rate = learning_rate


    def step( self, params, grads, lambd_factor = 0):
        L = len(params) // 2
    
        for l in reversed( range( 1, L+1)):
            dW = grads["dW"+str(l)]
            if 0 != lambd_factor:
                dW = dW + lambd_factor * params["W"+str(l)]
            
            params["W"+str(l)] = params["W"+str(l)] - self._learning_rate * dW
            params["b"+str(l)] = params["b"+str(l)] - self._learning_rate * grads["db"+str(l)]
            
        return params