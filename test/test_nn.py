import matplotlib.pyplot as plt
import time

import numpy as np

import unagi

from unagi.utils import load_data


#
# Train

# dataset shape = ( 250, 64, 64, 3) 250 samples, images: 64x64x3
train_x_orig, train_y, test_x_orig, test_y, classes = load_data( 
        '../datasets/train_catvnoncat.h5', '../datasets/test_catvnoncat.h5')

# Reshape the training and test examples 
train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0], -1).T   # The "-1" makes reshape flatten the remaining dimensions
test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0], -1).T

# Standardize data to have feature values between 0 and 1.
train_x = train_x_flatten/255.
test_x = test_x_flatten/255.

tic = time.time()

#Hyperparameters (most -> less important)
learning_rate = 0.0075
#Beta = 0.9
layers = [12288, 20, 7, 5, 1]
batch_size = 32
weight_decay = 0.7
drop_out = 0
#learning_rate_decay = 1
#Beta1, Beta2, Epsilon
#^

model = unagi.nn( layers)

model.set_seed(1)

model.optimizer.set_learning_rate( learning_rate)

#model.set_loss( Sigmoid_Cross_Entropy())

#model.setInitialization(Unagi.initialization.xavier())
#model.setOptimizer( Unagi.optimizer.GradientDescent(learning_rate = 0.0075))
model.train( train_x,
             train_y, 
  	     epochs = 2500,
             batch_size = batch_size,
             lambd = weight_decay,
             drop_out = drop_out)

print("Total training time: {0:.3f} secs".format(time.time()-tic))


# plot the cost
plt.plot(np.squeeze(model.costs))
plt.ylabel('cost')
plt.xlabel('iterations (per tens)')
plt.title("Learning rate =" + str(learning_rate))
plt.show()


#
# Test
#
pred_train = model.predict(train_x)
print("Accuracy Train: "  + str(np.sum((pred_train == train_y)/train_y.shape[1])))

pred_train = model.predict(test_x)
print("Accuracy Test: "  + str(np.sum((pred_train == test_y)/test_y.shape[1])))

