# An attempt to use a Recurrent Neural Network to learn and predict drone dynamics. In this version, the
# network inputs are the four motor commands and the outputs are the roll, pitch, and yaw angles.

from __future__ import division, print_function, absolute_import

import tflearn
from tflearn.layers.normalization import batch_normalization
import numpy as np
import pandas as pd
import math
import matplotlib
matplotlib.use('Agg')   # use('Agg') for saving to file and use('TkAgg') for interactive plot
import matplotlib.pyplot as plt

# Configuration Variables
input_dim = 4
output_dim = 3
steps_of_history = 10
batch_len = 128
epoch_len = 3

rawDataPath = 'DroneData/timeSeriesData.csv'

# Generate the data used in training
timeSeries = pd.read_csv(rawDataPath)
x = [timeSeries['m1CMD'], timeSeries['m2CMD'], timeSeries['m3CMD'], timeSeries['m4CMD']]
y = [timeSeries['pitch'], timeSeries['roll'], timeSeries['yaw']]

# Convert row vectors into column vectors
x = np.array(x);     y = np.array(y)
x = np.transpose(x); y = np.transpose(y)

# Generate the input and target training data
input_seq = []
output_seq = []

for i in range(0, len(timeSeries['rtosTick']) - steps_of_history):
    input_seq.append(x[i:i+steps_of_history, :])    # Time history input
    output_seq.append(y[i+steps_of_history, :])     # Single output resulting from ^^^


trainX = np.reshape(input_seq, [-1, input_dim, steps_of_history])
trainY = np.reshape(output_seq, [-1, output_dim])


# Build the network model
input_layer = tflearn.input_data(shape=[None, input_dim, steps_of_history])

layer1 = tflearn.simple_rnn(input_layer, n_units=10, activation='softmax', return_seq=True, name='Layer1')
layer2 = tflearn.simple_rnn(layer1, n_units=10, activation='sigmoid', name='Layer2')
layer3 = tflearn.fully_connected(layer2, output_dim, activation='linear', name='Layer3')

output_layer = tflearn.regression(layer3, optimizer='adam', loss='mean_square', learning_rate=0.1)


# Training
model = tflearn.DNN(output_layer, clip_gradients=0.3, tensorboard_verbose=0)
model.fit(trainX, trainY, n_epoch=epoch_len, validation_set=0.1, batch_size=batch_len)


# Generate a model prediction as a very simple sanity check...
predictY = model.predict(trainX)


# Plot the results
plt.figure(figsize=(20, 4))
plt.suptitle('Pitch Predictions')
plt.plot(trainY[:, 0], 'r-', label='Actual')
plt.plot(predictY[:, 0], 'g-', label='Predicted')
plt.legend()
plt.savefig('pitch.png')

plt.figure(figsize=(20, 4))
plt.suptitle('Roll Predictions')
plt.plot(trainY[:, 1], 'r-', label='Actual')
plt.plot(predictY[:, 1], 'g-', label='Predicted')
plt.legend()
plt.savefig('roll.png')

plt.figure(figsize=(20, 4))
plt.suptitle('Yaw Predictions')
plt.plot(trainY[:, 2], 'r-', label='Actual')
plt.plot(predictY[:, 2], 'g-', label='Predicted')
plt.legend()
plt.savefig('yaw.png')

