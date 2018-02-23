from __future__ import division, print_function, absolute_import

import tflearn
import tensorflow as tf
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
past_steps = 1250
batch_len = 128
epoch_len = 10
num_neurons_per_layer = 256
layer_dropout = (0.8, 0.8)

rawDataPath = '../DroneData/timeSeriesDataSmoothed.csv'
model_name = 'model.tfl'
ckpt_path = '../Checkpoints/DroneRNN_Ver2/' + model_name + '.ckpt'
best_ckpt_path = '../Checkpoints/DroneRNN_Ver2/BestResult/' + model_name + '.ckpt'


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
                                return_seq=True,
                                dropout=layer_dropout)

    layer3 = tflearn.simple_rnn(layer2,
                                n_units=neurons_per_layer,
                                activation='sigmoid',
                                dropout=layer_dropout)

    layer4 = tflearn.fully_connected(layer3,
                                     output_dim,
                                     activation='linear')

    output_layer = tflearn.regression(layer4,
                                      optimizer='adam',
                                      loss='mean_square',
                                      learning_rate=0.002)

    return tflearn.DNN(output_layer,
                       tensorboard_verbose=3,
                       checkpoint_path=ckpt_path,
                       best_checkpoint_path=best_ckpt_path)


def drone_lstm_model(input_dim, output_dim, steps_of_history, neurons_per_layer):
    input_layer = tflearn.input_data(shape=[None, input_dim, steps_of_history])

    layer1 = tflearn.lstm(input_layer,
                          n_units=neurons_per_layer,
                          return_seq=True,      # Need 3D Tensor input to layer 2
                          dropout=layer_dropout)

    layer2 = tflearn.lstm(layer1,
                          n_units=neurons_per_layer,
                          return_seq=True,      # Need 3D Tensor input to layer 3
                          dropout=layer_dropout)

    layer3 = tflearn.lstm(layer2,
                          n_units=neurons_per_layer,
                          return_seq=False,     # Need 2D Tensor input to layer 4
                          dropout=layer_dropout)

    layer4 = tflearn.fully_connected(layer3, n_units=output_dim)

    output_layer = tflearn.regression(layer4,
                                      optimizer='adam',
                                      loss='mean_square',
                                      learning_rate=0.001)

    return tflearn.DNN(output_layer,
                       tensorboard_verbose=3,
                       checkpoint_path=ckpt_path,
                       best_checkpoint_path=best_ckpt_path)


if parse_byte_data:
    parse_drone_logs()


tflearn.init_graph(num_cores=16, gpu_memory_fraction=0.8)
with tf.device('/gpu:0'):
    timeSeries = pd.read_csv(rawDataPath)
    input = np.array([timeSeries['m1CMD'], timeSeries['m2CMD'], timeSeries['m3CMD'], timeSeries['m4CMD']])
    output = np.array([timeSeries['pitch'], timeSeries['roll'], timeSeries['yaw']])

    trainX = []
    trainY = []

    for i in range(0, len(input[0, :]) - 2 * past_steps):
        trainX.append(input[0:input_size, i:i + past_steps])
        trainY.append(output[0:output_size, i + past_steps])

    trainX = np.reshape(trainX, [-1, input_size, past_steps])
    trainY = np.reshape(trainY, [-1, output_size])

    model = drone_rnn_model(input_size, output_size, past_steps, num_neurons_per_layer)

#"""
    model.fit(trainX, trainY,
              n_epoch=epoch_len,
              validation_set=0.25,
              batch_size=batch_len,
              show_metric=True,
              snapshot_epoch=True,
              run_id='model_and_weights')
# """


model.save(model_name)

model.load(model_name)

# Generate some data to use for prediction
trainX = []
trainY = []

for i in range(len(input[0, :]) - 2*past_steps, len(input[0, :])-past_steps):
    trainX.append(input[0:input_size, i:i + past_steps])
    trainY.append(output[0:output_size, i + past_steps])

X = np.reshape(trainX, [-1, input_size, past_steps])
Y = np.reshape(trainY, [-1, output_size])

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