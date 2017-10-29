import unagi.utils as utils
import numpy as np

from .loss import Loss

class Sigmoid_Cross_Entropy( Loss):
    def activation(self, Z):
        return utils.sigmoid( Z)


    def cost(self, logits, labels, params, lambd_factor):
        m = labels.shape[1]

        logprobs = np.multiply(-np.log(logits),labels) + np.multiply(-np.log(1 - logits), 1 - labels)
        cost = 1./m * np.sum(logprobs) + self.L2_Regularize(params, lambd_factor)

        return cost
