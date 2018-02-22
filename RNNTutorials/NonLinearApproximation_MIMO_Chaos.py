from __future__ import division, print_function, absolute_import

import tflearn
from tflearn.layers.normalization import batch_normalization
import numpy as np
import pandas as pd
import math
import matplotlib
matplotlib.use('TkAgg')   # use('Agg') for saving to file and use('TkAgg') for interactive plot
import matplotlib.pyplot as plt


input = np.r_[np.array(pd.read_csv('mackeyglass/mackey_glass1.csv')).transpose(),
              np.array(pd.read_csv('mackeyglass/mackey_glass2.csv')).transpose(),
              np.array(pd.read_csv('mackeyglass/mackey_glass3.csv')).transpose(),
              np.array(pd.read_csv('mackeyglass/mackey_glass4.csv')).transpose()]

# Configuration Variables
input_dim = 4                   # Number of parameters the NN will use as input
output_dim = 3                  # Number of NN outputs
steps_of_history = 2000         # Selects how large of a sample set will be used to train
batch_len = 128
epoch_len = 15
neurons_per_layer = 128
layer_dropout = (1.0, 1.0)

# Generate some noise to add on top of the input signal
input += np.random.random(np.shape(input)) * 0.1

# Weird and random non-linear functions for the NN to learn
output1 = np.tan(input[0, :] + np.cos(input[1, :]) - np.tanh(input[0, :])) - 0.5
output2 = np.cos(input[1, :] + np.cos(input[0, :]) - np.sin(input[2, :])*np.cos(input[3, :]))
output3 = np.sin(input[3, :] + input[2, :])

output = np.r_[output1[None, :],
               output2[None, :],
               output3[None, :]]

print(np.shape(output))

plt.figure(figsize=(16, 4))
plt.suptitle('Output of Non-Linear Function 1')
plt.plot(output[0, :], 'g-', label='Output1')
plt.legend()

plt.figure(figsize=(16, 4))
plt.suptitle('Output of Non-Linear Function 2')
plt.plot(output[1, :], 'g-', label='Output2')
plt.legend()

plt.figure(figsize=(16, 4))
plt.suptitle('Output of Non-Linear Function 2')
plt.plot(output[2, :], 'g-', label='Output2')
plt.legend()

# plt.show()

# Generate the input and training data
input_seq = []
output_seq = []

for i in range(0, len(input[0, :]) - steps_of_history):
    input_seq.append(input[:, i:i+steps_of_history])    # NOTE: Select all columns for multiple inputs...
    output_seq.append(output[:, i+steps_of_history])    # NOTE: Select all columns for multiple outputs...

trainX = np.reshape(input_seq, [-1, input_dim, steps_of_history])
trainY = np.reshape(output_seq, [-1, output_dim])

print(np.shape(trainX))
print(np.shape(trainY))

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

print(np.shape(trainX))
print(np.shape(predictY))

# Plot the results
plt.figure(figsize=(16, 4))
plt.suptitle('Function 1 Train vs Predict')
plt.plot(trainY[:, 0], 'r-', label='Actual')
plt.plot(predictY[:, 0], 'b-', label='Predicted')
plt.legend()

plt.figure(figsize=(16, 4))
plt.suptitle('Function 2 Train vs Predict')
plt.plot(trainY[:, 1], 'r-', label='Actual')
plt.plot(predictY[:, 1], 'b-', label='Predicted')
plt.legend()

plt.figure(figsize=(16, 4))
plt.suptitle('Function 3 Train vs Predict')
plt.plot(trainY[:, 2], 'r-', label='Actual')
plt.plot(predictY[:, 2], 'b-', label='Predicted')
plt.legend()

plt.show()



