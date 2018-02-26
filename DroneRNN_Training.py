from __future__ import division, print_function, absolute_import

import src.TensorFlowModels as TFModels

import tflearn
import tensorflow as tf
from Scripts import RawData2CSV as DataParser
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')   # use('Agg') for saving to file and use('TkAgg') for interactive plot
import matplotlib.pyplot as plt


if __name__ == "__main__":
    # Configuration Variables
    parse_byte_data = False     # Reconstructs the input data for training

    cfg = TFModels.ModelConfig()
    cfg.model_name = 'euler_model.tfl'
    cfg.input_size = 6
    cfg.input_depth = 10
    cfg.output_size = 2
    cfg.batch_len = 128
    cfg.epoch_len = 1
    cfg.neurons_per_layer = 128
    cfg.layer_dropout = (0.8, 0.8)

    rawDataPath     = 'DroneData/csv/timeSeriesDataSmoothed.csv'
    ckpt_path       = 'Checkpoints/DroneRNN_Ver3/EulerModel' + cfg.model_name + '.ckpt'
    best_ckpt_path  = 'Checkpoints/DroneRNN_Ver3/EulerModel/BestResult/' + cfg.model_name + '.ckpt'
    cfg_path        = 'euler_model_cfg.csv'

    cfg.save(cfg_path)

    # ---------------------------------------
    # Update the input data into the model for training
    # ---------------------------------------
    if parse_byte_data:
        DataParser.raw_data_2_csv()
        DataParser.create_time_series_from_csv_logs()
        DataParser.smooth_time_series_data()

    # ---------------------------------------
    # Train the Euler Prediction Network
    # ---------------------------------------
    tflearn.init_graph(num_cores=16, gpu_memory_fraction=0.8)
    with tf.device('/gpu:0'):
        timeSeries = pd.read_csv(rawDataPath)

        input = np.array([timeSeries['m1CMD'], timeSeries['m2CMD'], timeSeries['m3CMD'], timeSeries['m4CMD'],
                          timeSeries['asp'], timeSeries['asr']])

        output = np.array([timeSeries['pitch'], timeSeries['roll']])

        input = input.transpose()
        output = output.transpose()

        trainX = []
        trainY = []

        trainLen = len(input[:, 0]) - 10 * cfg.input_depth

        for i in range(0, trainLen):
            trainX.append(input[i:i + cfg.input_depth, 0:cfg.input_size])
            trainY.append(output[i + cfg.input_depth, 0:cfg.output_size])

        trainX = np.reshape(trainX, [-1, cfg.input_depth, cfg.input_size])
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

        print("Deleting training data")
        del timeSeries
        del trainX
        del trainY
    # ---------------------------------------
    # Plot some data for the user to see how well training went
    # ---------------------------------------
    X = []
    Y = []

    for i in range(len(input[:, 0]) - trainLen, len(input[:, 0])-cfg.input_depth):
        X.append(input[i:i + cfg.input_depth, 0:cfg.input_size])
        Y.append(output[i + cfg.input_depth, 0:cfg.output_size])

    X = np.reshape(X, [-1, cfg.input_depth, cfg.input_size])
    Y = np.reshape(Y, [-1, cfg.output_size])

    predictY = model.predict(X)

    # Plot the results
    print("Plotting Sample Outputs")
    imgSavePath = "Checkpoints/DroneRNN_Ver3/EulerModel/Images/"

    # PITCH
    plt.figure(figsize=(16, 4))
    plt.suptitle('Pitch Actual Vs Predicted')
    plt.plot(Y[:, 0], 'r-', label='Actual')
    plt.plot(predictY[:, 0], 'g-', label='Predicted')
    plt.legend()
    plt.savefig(imgSavePath+'pitch.png')

    # ROLL
    plt.figure(figsize=(16, 4))
    plt.suptitle('Roll Actual vs Predicted')
    plt.plot(Y[1, :], 'r-', label='Actual')
    plt.plot(predictY[1, :], 'g-', label='Predicted')
    plt.legend()
    plt.savefig(imgSavePath + 'roll.png')



    # ---------------------------------------
    # Train the Gyro Prediction Network
    # ---------------------------------------
    with tf.device('/gpu:0'):
        cfg = TFModels.ModelConfig()
        cfg.model_name = 'gyro_model.tfl'
        cfg.input_size = 6
        cfg.input_depth = 50
        cfg.output_size = 3
        cfg.batch_len = 128
        cfg.epoch_len = 1
        cfg.neurons_per_layer = 32
        cfg.layer_dropout = (0.8, 0.8)

        rawDataPath = 'DroneData/csv/timeSeriesDataSmoothed.csv'
        ckpt_path = 'Checkpoints/DroneRNN_Ver3/GyroModel/' + cfg.model_name + '.ckpt'
        best_ckpt_path = 'Checkpoints/DroneRNN_Ver3/GyroModel/BestResult/' + cfg.model_name + '.ckpt'
        cfg_path = 'gyro_model_cfg.csv'

        cfg.save(cfg_path)

        timeSeries = pd.read_csv(rawDataPath)

        input = np.array([timeSeries['m1CMD'], timeSeries['m2CMD'], timeSeries['m3CMD'], timeSeries['m4CMD'],
                          timeSeries['asp'], timeSeries['asr']])

        output = np.array([timeSeries['gx'], timeSeries['gy'], timeSeries['gz']])

        input = input.transpose()
        output = output.transpose()

        trainX = []
        trainY = []

        trainLen = len(input[:, 0]) - 10 * cfg.input_depth

        for i in range(0, trainLen):
            trainX.append(input[i:i + cfg.input_depth, 0:cfg.input_size])
            trainY.append(output[i + cfg.input_depth, 0:cfg.output_size])

        trainX = np.reshape(trainX, [-1, cfg.input_depth, cfg.input_size])
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

        print("Deleting training data")
        del timeSeries
        del trainX
        del trainY
    # ---------------------------------------
    # Plot some data for the user to see how well training went
    # ---------------------------------------
    X = []
    Y = []

    for i in range(len(input[:, 0]) - trainLen, len(input[:, 0]) - cfg.input_depth):
        X.append(input[i:i + cfg.input_depth, 0:cfg.input_size])
        Y.append(output[i + cfg.input_depth, 0:cfg.output_size])

    X = np.reshape(X, [-1, cfg.input_depth, cfg.input_size])
    Y = np.reshape(Y, [-1, cfg.output_size])

    predictY = model.predict(X)

    # Plot the results
    print("Plotting Sample Outputs")
    imgSavePath = "Checkpoints/DroneRNN_Ver3/GyroModel/Images/"

    # GYRO X
    plt.figure(figsize=(16, 4))
    plt.suptitle('GX Actual vs Predicted')
    plt.plot(Y[2, :], 'r-', label='Actual')
    plt.plot(predictY[2, :], 'g-', label='Predicted')
    plt.legend()
    plt.savefig(imgSavePath + 'gx.png')

    # GYRO Y
    plt.figure(figsize=(16, 4))
    plt.suptitle('GY Actual vs Predicted')
    plt.plot(Y[3, :], 'r-', label='Actual')
    plt.plot(predictY[3, :], 'g-', label='Predicted')
    plt.legend()
    plt.savefig(imgSavePath + 'gy.png')

    # GYRO Z
    plt.figure(figsize=(16, 4))
    plt.suptitle('GZ Actual vs Predicted')
    plt.plot(Y[4, :], 'r-', label='Actual')
    plt.plot(predictY[4, :], 'g-', label='Predicted')
    plt.legend()
    plt.savefig(imgSavePath + 'gz.png')