import numpy as np

from nn_2L import nn_model
from nn_2L import predict

# 
# Train
#

X_train = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]]).T
        
Y_OR = np.array([[0,1,1,1]])
Y_AND = np.array([[0,0,0,1]])
Y_XOR = np.array([[0,1,1,0]])

Y_train = Y_XOR
        
params = nn_model( X_train, Y_train, hidden_layer = 4, learning_rate = 1.2)


#
# Predict
#

print("Ground truth: {}\nPrediction: {}".format(Y_train, predict(X_train, params)))


