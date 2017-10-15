import numpy as np
import matplotlib.pyplot as plt
import time

from nn_utils import load_data
from nn import nn_model, nn_predict

plt.rcParams['figure.figsize'] = (5.0, 4.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'white'


#
# Train
#

np.random.seed(1)

# dataset shape = ( 250, 64, 64, 3) 250 samples, images: 64x64x3
train_x_orig, train_y, test_x_orig, test_y, classes = load_data( 
        'datasets/train_catvnoncat.h5', 'datasets/test_catvnoncat.h5')

# Reshape the training and test examples 
train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0], -1).T   # The "-1" makes reshape flatten the remaining dimensions
test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0], -1).T

# Standardize data to have feature values between 0 and 1.
train_x = train_x_flatten/255.
test_x = test_x_flatten/255.

tic = time.time()

learning_rate = 0.0075
layers_dims = [12288, 20, 7, 5, 1] #  5-layer model
parameters, costs = nn_model(train_x, 
                             train_y, 
                             layers_dims, 
                             initialization = "xavier", 
                             epochs = 2500, 
                             lambd = 0.7,
                             drop_out = 0.1)

print("Total training time: {0:.3f} secs".format(time.time()-tic))

# plot the cost
plt.plot(np.squeeze(costs))
plt.ylabel('cost')
plt.xlabel('iterations (per tens)')
plt.title("Learning rate =" + str(learning_rate))
plt.show()


#
# Test
#
pred_train = nn_predict(train_x, parameters)
print("Accuracy Train: "  + str(np.sum((pred_train == train_y)/train_y.shape[1])))

pred_train = nn_predict(test_x, parameters)
print("Accuracy Test: "  + str(np.sum((pred_train == test_y)/test_y.shape[1])))