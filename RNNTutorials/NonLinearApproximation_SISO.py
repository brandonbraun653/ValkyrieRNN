from __future__ import division, print_function, absolute_import

import tflearn
from tflearn.layers.normalization import batch_normalization
import numpy as np
import pandas as pd
import math
import matplotlib
matplotlib.use('TkAgg')   # use('Agg') for saving to file and use('TkAgg') for interactive plot
import matplotlib.pyplot as plt

# Configuration Variables
input_dim = 1
output_dim = 1
steps_of_history = 2000
batch_len = 128
epoch_len = 15
neurons_per_layer = 128
layer_dropout = (1.0, 1.0)
step_radians = 0.01


# Generate some non-linear IO
input = np.sin(np.arange(0, 20*math.pi, step_radians))
randomVal = np.random.random(np.shape(input)) * 0.1
input = input + randomVal
output = np.tan(input + np.cos(input) - np.tan(input)) - 0.5

plt.figure(figsize=(20, 4))
plt.plot(input, 'r-', label='Input')
plt.plot(output, 'g-', label='Output')
plt.legend()
# plt.show()

# Generate the input and training data
input_seq = []
output_seq = []

for i in range(0, len(input) - steps_of_history):
    input_seq.append(input[i:i+steps_of_history])
    output_seq.append(output[i+steps_of_history])

trainX = np.reshape(input_seq, [-1, input_dim, steps_of_history])
trainY = np.reshape(output_seq, [-1, output_dim])


# Build the network model
input_layer = tflearn.input_data(shape=[None, input_dim, steps_of_history])

layer1 = tflearn.simple_rnn(input_layer,
                            n_units=neurons_per_layer,
                            activation='relu',
                            return_seq=True,
                            dropout=layer_dropout,
                            name='Layer1')
layer2 = tflearn.simple_rnn(layer1,
                            n_units=neurons_per_layer,
                            activation='sigmoid',
                            dropout=layer_dropout,
                            name='Layer2')

layer3 = tflearn.fully_connected(layer2,
                                 output_dim,
                                 activation='linear',
                                 name='Layer3')

output_layer = tflearn.regression(layer3,
                                  optimizer='adam',
                                  loss='mean_square',
                                  learning_rate=0.002)


# Training
model = tflearn.DNN(output_layer, clip_gradients=0.1, tensorboard_verbose=3)
model.fit(trainX, trainY, n_epoch=epoch_len, validation_set=0.1, batch_size=batch_len)

# Generate a model prediction as a very simple sanity check...
predictY = model.predict(trainX)

# Plot the results
plt.figure(figsize=(20, 4))
plt.plot(trainY, 'r-', label='Actual')
plt.plot(predictY, 'g-', label='Predicted')
plt.legend()
plt.show()



