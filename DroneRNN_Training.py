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
    train_euler_model = False
    train_gyro_model = True

    max_cpu_cores = 4
    max_gpu_mem = 0.5

    # ---------------------------------------
    # Update the input data into the model for training
    # ---------------------------------------
    if False:
        DataParser.raw_data_2_csv()
        DataParser.create_time_series_from_csv_logs()
        DataParser.smooth_time_series_data()

    # ---------------------------------------
    # Train the Euler Prediction Network
    # ---------------------------------------
    if train_euler_model:
        # Generate the model configuration
        cfg = TFModels.ModelConfig()
        cfg.model_name = 'euler_model.tfl'
        cfg.input_size = 6
        cfg.input_depth = 500
        cfg.output_size = 2
        cfg.batch_len = 128
        cfg.epoch_len = 10
        cfg.neurons_per_layer = 32
        cfg.layer_dropout = (0.8, 0.8)
        cfg.save('euler_model_cfg.csv')
        train_data_len = 25 * 1000      # TODO: Should this be put in the cfg object?

        # Add a few configurations for saving the model at various stages in training
        rawDataPath     = 'DroneData/csv/timeSeriesDataSmoothed.csv'
        ckpt_path       = 'Checkpoints/DroneRNN_Ver3/EulerModel/' + cfg.model_name + '.ckpt'
        best_ckpt_path  = 'Checkpoints/DroneRNN_Ver3/EulerModel/BestResult/' + cfg.model_name + '.ckpt'
        last_ckpt_path  = 'Checkpoints/DroneRNN_Ver3/EulerModel/LastResult/' + cfg.model_name + '.ckpt'

        # Now actually do the training
        print("STARTING TRAINING OF EULER MODEL")
        tflearn.init_graph(num_cores=max_cpu_cores, gpu_memory_fraction=max_gpu_mem)
        with tf.device('/gpu:0'):
            timeSeries = pd.read_csv(rawDataPath)

            input_full = np.array([timeSeries['m1CMD'],
                                   timeSeries['m2CMD'],
                                   timeSeries['m3CMD'],
                                   timeSeries['m4CMD'],
                                   timeSeries['asp'],
                                   timeSeries['asr']]).transpose()

            output_full = np.array([timeSeries['pitch'],
                                    timeSeries['roll']]).transpose()

            num_samples = np.shape(input_full)
            total_train_iterations = int(np.ceil(num_samples[0] / train_data_len))

            # Create the basic model to train with iteratively
            model = TFModels.drone_rnn_model(dim_in=cfg.input_size,
                                             dim_out=cfg.output_size,
                                             past_depth=cfg.input_depth,
                                             layer_neurons=cfg.neurons_per_layer,
                                             layer_dropout=cfg.layer_dropout,
                                             learning_rate=cfg.learning_rate,
                                             checkpoint_path=ckpt_path,
                                             best_checkpoint_path=best_ckpt_path)

            # Train the above model with the full dataset being broken up into
            # multiple parts to handle RAM overload
            current_train_idx = 0
            trainX = []
            trainY = []

            for train_iteration in range(0, total_train_iterations):
                # Reset the training data for a new batch
                trainX = []
                trainY = []

                # Grab a full set of data if we have enough left
                if (current_train_idx + train_data_len) < num_samples[0]:

                    for i in range(0, train_data_len):
                        i = current_train_idx + i

                        trainX.append(input_full[i:i + cfg.input_depth, 0:cfg.input_size])
                        trainY.append(output_full[i + cfg.input_depth, 0:cfg.output_size])

                # Otherwise, only grab the remaining data we can fit
                else:
                    for i in range(0, (num_samples[0] - current_train_idx - cfg.input_depth)):
                        i = i + current_train_idx

                        trainX.append(input_full[i:i + cfg.input_depth, 0:cfg.input_size])
                        trainY.append(output_full[i + cfg.input_depth, 0:cfg.output_size])

                # Reshape for NN input (X)[batch_size, sample_depth, sample_values], (Y)[batch_size, output_values]
                trainX = np.reshape(trainX, [-1, cfg.input_depth, cfg.input_size])
                trainY = np.reshape(trainY, [-1, cfg.output_size])

                current_train_idx += train_data_len

                model.fit(trainX, trainY,
                          n_epoch=cfg.epoch_len,
                          validation_set=0.25,
                          batch_size=cfg.batch_len,
                          show_metric=True,
                          snapshot_epoch=True,
                          run_id='HereGoesNothing')

            # Save the last state of the model to a known location with a known name for quick
            # lookup in the inferencing script. This may or many not be the best performer.
            model.save(last_ckpt_path)

    # ---------------------------------------
    # Train the Gyro Prediction Network
    # ---------------------------------------
    if train_gyro_model:
        # Generate the model configuration
        cfg = TFModels.ModelConfig()
        cfg.model_name = 'gyro_model.tfl'
        cfg.input_size = 6
        cfg.input_depth = 500
        cfg.output_size = 3
        cfg.batch_len = 128
        cfg.epoch_len = 10
        cfg.neurons_per_layer = 32
        cfg.layer_dropout = (0.8, 0.8)
        cfg.save('gyro_model_cfg.csv')
        train_data_len = 25 * 1000  # TODO: Should this be put in the cfg object?

        # Add a few configurations for saving the model at various stages in training
        rawDataPath = 'DroneData/csv/timeSeriesDataSmoothed.csv'
        ckpt_path = 'Checkpoints/DroneRNN_Ver3/GyroModel/' + cfg.model_name + '.ckpt'
        best_ckpt_path = 'Checkpoints/DroneRNN_Ver3/GyroModel/BestResult/' + cfg.model_name + '.ckpt'
        last_ckpt_path = 'Checkpoints/DroneRNN_Ver3/GyroModel/LastResult/' + cfg.model_name + '.ckpt'

        # Now actually do the training
        print("STARTING TRAINING OF GYRO MODEL")
        tflearn.init_graph(num_cores=max_cpu_cores, gpu_memory_fraction=max_gpu_mem)
        with tf.device('/gpu:0'):
            timeSeries = pd.read_csv(rawDataPath)

            input_full = np.array([timeSeries['m1CMD'],
                                   timeSeries['m2CMD'],
                                   timeSeries['m3CMD'],
                                   timeSeries['m4CMD'],
                                   timeSeries['asp'],
                                   timeSeries['asr']]).transpose()

            output_full = np.array([timeSeries['gx'],
                                    timeSeries['gy'],
                                    timeSeries['gz']]).transpose()

            num_samples = np.shape(input_full)
            total_train_iterations = int(np.ceil(num_samples[0] / train_data_len))

            model = TFModels.drone_rnn_model(dim_in=cfg.input_size,
                                             dim_out=cfg.output_size,
                                             past_depth=cfg.input_depth,
                                             layer_neurons=cfg.neurons_per_layer,
                                             layer_dropout=cfg.layer_dropout,
                                             learning_rate=cfg.learning_rate,
                                             checkpoint_path=ckpt_path,
                                             best_checkpoint_path=best_ckpt_path)

            # Train the above model with the full dataset being broken up into
            # multiple parts to handle RAM overload
            current_train_idx = 0
            trainX = []
            trainY = []

            for train_iteration in range(0, total_train_iterations):
                # Reset the training data for a new batch
                trainX = []
                trainY = []

                # Grab a full set of data if we have enough left
                if (current_train_idx + train_data_len) < num_samples[0]:

                    for i in range(0, train_data_len):
                        i = current_train_idx + i

                        trainX.append(input_full[i:i + cfg.input_depth, 0:cfg.input_size])
                        trainY.append(output_full[i + cfg.input_depth, 0:cfg.output_size])

                # Otherwise, only grab the remaining data we can fit
                else:
                    for i in range(0, (num_samples[0] - current_train_idx - cfg.input_depth)):
                        i = i + current_train_idx

                        trainX.append(input_full[i:i + cfg.input_depth, 0:cfg.input_size])
                        trainY.append(output_full[i + cfg.input_depth, 0:cfg.output_size])

                # Reshape for NN input (X)[batch_size, sample_depth, sample_values], (Y)[batch_size, output_values]
                trainX = np.reshape(trainX, [-1, cfg.input_depth, cfg.input_size])
                trainY = np.reshape(trainY, [-1, cfg.output_size])

                current_train_idx += train_data_len

                model.fit(trainX, trainY,
                          n_epoch=cfg.epoch_len,
                          validation_set=0.25,
                          batch_size=cfg.batch_len,
                          show_metric=True,
                          snapshot_epoch=True,
                          run_id='HereGoesNothing')

            # Save the last state of the model to a known location with a known name for quick
            # lookup in the inferencing script. This may or many not be the best performer.
            model.save(last_ckpt_path)

