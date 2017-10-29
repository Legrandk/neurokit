import unittest as ut

import numpy as np

import unagi
from unagi.utils import params_to_vector, vector_to_parameters, grads_to_vector

class TestGrads( ut.TestCase):
    def test_grads(self):
        np.random.seed(1)

        M = 16

        X = np.random.randn(4, M)
        Y = np.random.randn(1, M) < 0.7

        learning_rate = 0.0075
        layers_dims = [4, 3, 3, 1]
        lambd = 0 #0.7
        drop_out = 0

        model = unagi.nn( layers_dims, initializer="he")
        model.train( X, Y, 
                     epochs = 1, batch_size = M, lambd = lambd, drop_out = drop_out)
        parameters = model.parameters

        lambd_factor = lambd / M
        A, caches = model.forward_propagation( X, parameters)
        J = model.compute_cost( A, Y, parameters, lambd_factor)
        grads,_ = model.backward_propagation( A, Y, parameters, caches, lambd_factor, drop_out = drop_out)

        epsilon = 1e-7
        p_values, p_shape = params_to_vector( parameters)
        grads_values,_    =  grads_to_vector( grads)

        grads_approx = np.zeros(grads_values.shape)

        inv_2e = 1. / (2*epsilon)

        num_params = p_values.shape[0]
        for i in range(0, num_params):
            theta_plus = np.copy(p_values)
            theta_plus[i,0] = theta_plus[i,0] + epsilon

            theta_minus = np.copy(p_values)
            theta_minus[i,0] = theta_minus[i,0] - epsilon
            
            AL,_ = model.forward_propagation( X, vector_to_parameters(theta_plus, p_shape))
            J_plus = model.compute_cost( AL, Y, vector_to_parameters(theta_plus, p_shape), lambd_factor)
            
            AL,_ = model.forward_propagation( X, vector_to_parameters(theta_minus, p_shape))
            J_minus = model.compute_cost( AL, Y, vector_to_parameters(theta_minus, p_shape), lambd_factor)

            grads_approx[i] = inv_2e * (J_plus - J_minus)

        numerator = np.linalg.norm( grads_approx - grads_values)
        denominator = np.linalg.norm( grads_approx) + np.linalg.norm(grads_values)

        diff = numerator / denominator

        #print("Diff: {}".format(diff))
        self.assertTrue( diff <= 1e-7, "*** WARNING: There is a mistake in the backward propagation! difference = "+repr(diff))

if __name__ == '__main__':
    ut.main()
