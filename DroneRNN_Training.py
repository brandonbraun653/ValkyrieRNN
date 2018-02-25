from __future__ import division, print_function, absolute_import

import src.TensorFlowModels as TFModels

import tflearn
import tensorflow as tf
from Scripts import RawData2CSV as DataParser
import numpy as np
import pandas as pd
import math
import matplotlib
matplotlib.use('TkAgg')   # use('Agg') for saving to file and use('TkAgg') for interactive plot
import matplotlib.pyplot as plt


if __name__ == "__main__":
    # Configuration Variables
    parse_byte_data = True     # Reconstructs the input data for training

    cfg = TFModels.ModelConfig()
    cfg.model_name = 'model.tfl'
    cfg.input_size = 6
    cfg.input_depth = 1000
    cfg.output_size = 5
    cfg.batch_len = 128
    cfg.epoch_len = 3
    cfg.neurons_per_layer = 256
    cfg.layer_dropout = (0.8, 0.8)

    rawDataPath     = 'DroneData/csv/timeSeriesInterpolated.csv'
    ckpt_path       = 'Checkpoints/DroneRNN_Ver3/' + cfg.model_name + '.ckpt'
    best_ckpt_path  = 'Checkpoints/DroneRNN_Ver3/BestResult/' + cfg.model_name + '.ckpt'
    cfg_path        = 'model_cfg.csv'

    cfg.save(cfg_path)

    # Update the input data into the model for training
    if parse_byte_data:
        DataParser.raw_data_2_csv()
        DataParser.create_time_series_from_csv_logs()

    # ---------------------------------------
    # Train a new network
    # ---------------------------------------
    tflearn.init_graph(num_cores=16, gpu_memory_fraction=0.8)

    with tf.device('/gpu:0'):
        timeSeries = pd.read_csv(rawDataPath)

        input = np.array([timeSeries['m1CMD'], timeSeries['m2CMD'], timeSeries['m3CMD'], timeSeries['m4CMD'],
                          timeSeries['asp'], timeSeries['asr']])

        output = np.array([timeSeries['pitch'], timeSeries['roll'],
                           timeSeries['gx'], timeSeries['gy'], timeSeries['gz']])

        trainX = []
        trainY = []

        trainLen = len(input[0, :]) - 4 * cfg.input_depth

        for i in range(0, trainLen):
            trainX.append(input[0:cfg.input_size, i:i + cfg.input_depth])
            trainY.append(output[0:cfg.output_size, i + cfg.input_depth])

        trainX = np.reshape(trainX, [-1, cfg.input_size, cfg.input_depth])
        trainY = np.reshape(trainY, [-1, cfg.output_size])

        model = TFModels.drone_rnn_model(dim_in=cfg.input_size,
                                         dim_out=cfg.output_size,
                                         past_depth=cfg.input_depth,
                                         layer_neurons=cfg.neurons_per_layer,
                                         layer_dropout=cfg.layer_dropout,
                                         learning_rate=cfg.learning_rate,
                                         checkpoint_path=ckpt_path,
                                         best_checkpoint_path=best_ckpt_path)

        model.fit(trainX, trainY,
                  n_epoch=cfg.epoch_len,
                  validation_set=0.25,
                  batch_size=cfg.batch_len,
                  show_metric=True,
                  snapshot_epoch=True,
                  run_id='HereGoesNothing')

        model.save(cfg.model_name)

    # Plot some data for the user to see how well training went
    X = []
    Y = []

    for i in range(len(input[0, :]) - trainLen, len(input[0, :])-cfg.input_depth):
        X.append(input[0:cfg.input_size, i:i + cfg.input_depth])
        Y.append(output[0:cfg.output_size, i + cfg.input_depth])

    X = np.reshape(X, [-1, cfg.input_size, cfg.input_depth])
    Y = np.reshape(Y, [-1, cfg.output_size])

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