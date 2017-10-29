import numpy as np

class Loss:
    def L2_Regularize( self, params, lambd_factor = 0):
        l2_reg_cost = 0
        if 0 != lambd_factor: # lambd_factor = lambd / m
            L = len(params) // 2
            for l in range(1,L+1):
                l2_reg_cost += np.sum(np.square(params["W"+str(l)]))
            l2_reg_cost = (lambd_factor*0.5) * l2_reg_cost

        return l2_reg_cost

    def activation( self, Z):
        pass

    def cost(self, logits, labels, params, lambd_factor):
        pass

    def derivative(self, labels, logits):
        return logits - labels