from __future__ import division, print_function, absolute_import

import tflearn
from tflearn.layers.normalization import batch_normalization
from Scripts import RawData2CSV as DataParser
import numpy as np
import pandas as pd
import math
import matplotlib
matplotlib.use('TkAgg')   # use('Agg') for saving to file and use('TkAgg') for interactive plot
import matplotlib.pyplot as plt

# Configuration Variables
parse_byte_data = False      # Reconstructs the input data for training
train_new_model = True      # If true, retrains and saves the model. If false, loads the old model and runs it
input_size = 4
output_size = 2
past_steps = 2000
batch_len = 128
epoch_len = 15
num_neurons_per_layer = 128
layer_dropout = (1.0, 1.0)

rawDataPath = 'DroneData/timeSeriesDataSmoothed.csv'
model_name = './myModel'


def parse_drone_logs():
    DataParser.raw_data_2_csv()
    DataParser.create_time_series_from_csv_logs()
    DataParser.smooth_time_series_data()


def drone_rnn_model(input_dim, output_dim, steps_of_history, neurons_per_layer):

    input_layer = tflearn.input_data(shape=(None, input_dim, steps_of_history))

    layer1 = tflearn.simple_rnn(input_layer,
                                n_units=neurons_per_layer,
                                activation='relu',
                                return_seq=True,
                                dropout=layer_dropout)

    layer2 = tflearn.simple_rnn(layer1,
                                n_units=neurons_per_layer,
                                activation='sigmoid',
                                dropout=layer_dropout)

    layer3 = tflearn.fully_connected(layer2,
                                     output_dim,
                                     activation='linear')

    output_layer = tflearn.regression(layer3,
                                      optimizer='adam',
                                      loss='mean_square',
                                      learning_rate=0.002)

    return tflearn.DNN(output_layer, clip_gradients=0.1, tensorboard_verbose=3)


def train_rnn(save_name=None, x=0, y=0):
    model = drone_rnn_model(input_size, output_size, past_steps, num_neurons_per_layer)
    model.fit(x, y, n_epoch=epoch_len, validation_set=0.1, batch_size=batch_len)

    model.save(save_name)


if parse_byte_data:
    parse_drone_logs()


timeSeries = pd.read_csv(rawDataPath)
input = np.array([timeSeries['m1CMD'], timeSeries['m2CMD'], timeSeries['m3CMD'], timeSeries['m4CMD']])
output = np.array([timeSeries['pitch'], timeSeries['roll'], timeSeries['yaw']])


input_seq = []
output_seq = []

for i in range(0, len(input[0, :]) - 2*past_steps):
    input_seq.append(input[0:input_size, i:i + past_steps])
    output_seq.append(output[0:output_size, i + past_steps])

trainX = np.reshape(input_seq, [-1, input_size, past_steps])
trainY = np.reshape(output_seq, [-1, output_size])

# train_rnn(model_name, trainX, trainY)
model = drone_rnn_model(input_size, output_size, past_steps, num_neurons_per_layer)
model.fit(trainX, trainY, n_epoch=epoch_len, validation_set=0.1, batch_size=batch_len)


# model = drone_rnn_model(input_size, output_size, past_steps, num_neurons_per_layer)
# model.load(model_name, weights_only=True)

# Generate some data to use for prediction
input_seq = []
output_seq = []

for i in range(len(input[0, :]) - 2*past_steps, len(input[0, :])-past_steps):
    input_seq.append(input[0:input_size, i:i + past_steps])
    output_seq.append(output[0:output_size, i + past_steps])

X = np.reshape(input_seq, [-1, input_size, past_steps])
Y = np.reshape(output_seq, [-1, output_size])

predictY = model.predict(X)


# Plot the results
plt.figure(figsize=(16, 4))
plt.suptitle('Pitch Actual Vs Predicted')
plt.plot(Y[:, 0], 'r-', label='Actual')
plt.plot(predictY[:, 0], 'b-', label='Predicted')
plt.legend()

plt.figure(figsize=(16, 4))
plt.suptitle('Roll Actual vs Predicted')
plt.plot(Y[:, 1], 'r-', label='Actual')
plt.plot(predictY[:, 1], 'g-', label='Predicted')
plt.legend()

# plt.figure(figsize=(16, 4))
# plt.suptitle('Yaw Actual vs Predicted')
# plt.plot(trainY[:, 2], 'r-', label='Actual')
# plt.plot(predictY[:, 2], 'g-', label='Predicted')
# plt.legend()

plt.show()