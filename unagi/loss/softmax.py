import unagi.utils as utils
import numpy as np

from .loss import Loss

class Softmax_Cross_Entropy( Loss):
    def activation(self, Z):
        return utils.softmax(Z)


    def cost(self, logits, labels, params, lambd_factor):
        assert( 1 < logits.shape[0]), "Output dimension must be greater than 1"

        return -np.sum( labels*np.log(logits), axis = 0)
