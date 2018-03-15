# Simple example using recurrent neural networks to predict time series values
# This is going to be heavily commented to help me out as I am learning TF...

from __future__ import division, print_function, absolute_import

import tflearn
from tflearn.layers.normalization import batch_normalization
import numpy as np
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

step_radians = 0.01
steps_of_history = 200
steps_in_future = 1
index = 0
runTraining = True

# Raw signal we are trying to learn: 0-20pi dataset with 0.01 radian step
x = np.sin(np.arange(0, 20*math.pi, step_radians))

seq = []
next_val = []

# Generate raw training/prediction data. The idea is to create a
# "snapshot" of the past history (len(x) wide) and also store the
# next value in the sequence given the recorded snapshot.
# ie Input => Output, (Nx1) => (1x1)
for i in range(0, len(x) - steps_of_history, steps_in_future):
    seq.append(x[i: i + steps_of_history])      # Sliding window of past history data
    next_val.append(x[i + steps_of_history])    # Next value given ^^^


# Data is in a raw list format and needs to be reshaped
seq = np.reshape(seq, [-1, steps_of_history, 1])    # N x soh x 1
next_val = np.reshape(next_val, [-1, 1])            # N x 1
print(np.shape(seq))

trainX = np.array(seq)
trainY = np.array(next_val)

# Build the network:
# I think this is a kind of serialized operation where the output of one line
# is built upon by the next to gradually define the entire network structure
net = tflearn.input_data(shape=[None, steps_of_history, 1])
net = tflearn.simple_rnn(net, n_units=32, return_seq=False)
net = tflearn.fully_connected(net, 1, activation='linear')  # Why switch from rnn type? Is this an operator on net?
net = tflearn.regression(net, optimizer='Adam', loss='mean_square', learning_rate=0.1)


if runTraining:
    # Training
    model = tflearn.DNN(net, tensorboard_verbose=3)
    model.fit(trainX, trainY, n_epoch=15, validation_set=0.1, batch_size=128)

    # Test the trained model
    x = np.sin(np.arange(20*math.pi, 24*math.pi, step_radians))  # This goes an extra 4 pi steps beyond the initial set

    # Do the whole slicing thing again to generate testing data
    seq = []
    for i in range(0, len(x) - steps_of_history, steps_in_future):
        seq.append(x[i: i + steps_of_history])

    seq = np.reshape(seq, [-1, steps_of_history, 1])
    testX = np.array(seq)

    # Predict the future values
    predictY = model.predict(testX)
    print(np.shape(predictY))

    # Plot the results
    plt.figure(figsize=(20, 4))
    plt.suptitle('Prediction')
    plt.title('History=' + str(steps_of_history) + ', Future=' + str(steps_in_future))
    plt.plot(x, 'r-', label='Actual')
    plt.plot(predictY, 'gx', label='Predicted')
    plt.legend()
    plt.savefig('sine.png')
