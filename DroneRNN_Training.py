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
    train_small_for_testing = True
    reparse_input_data = False
    train_euler_model = False
    train_gyro_model = True

    max_cpu_cores = 16
    max_gpu_mem = 0.8
    max_input_depth = 1250

    if train_small_for_testing:
        max_input_depth = 100

    # ---------------------------------------
    # Update the input data into the model for training
    # ---------------------------------------
    if reparse_input_data:
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
        cfg.input_size = 4
        cfg.input_depth = max_input_depth
        cfg.output_size = 2
        cfg.batch_len = 128
        cfg.epoch_len = 10
        cfg.neurons_per_layer = 512
        cfg.layer_dropout = (0.8, 0.8)
        train_data_len = 75 * 1000  # TODO: Should this be put in the cfg object?

        # Add a few configurations for saving the model at various stages in training
        rawDataPath = 'DroneData/csv/timeSeriesDataSmoothed.csv'
        ckpt_path = 'Checkpoints/DroneRNN_Ver3/EulerModel/' + cfg.model_name + '.ckpt'
        best_ckpt_path = 'Checkpoints/DroneRNN_Ver3/EulerModel/BestResult/' + cfg.model_name + '.ckpt'
        last_ckpt_path = 'Checkpoints/DroneRNN_Ver3/EulerModel/LastResult/' + cfg.model_name + '.ckpt'

        if train_small_for_testing:
            cfg.epoch_len = 1
            cfg.neurons_per_layer = 32
            train_data_len = 10 * 1000
            last_ckpt_path = 'Checkpoints/DroneRNN_Ver3/EulerModel/SmallTesting/' + cfg.model_name + '.ckpt'

        cfg.save('euler_model_cfg.csv')



        # Now actually do the training
        print("STARTING TRAINING OF EULER MODEL")
        tf.reset_default_graph()
        with tf.variable_scope('euler'):
            tflearn.init_graph(num_cores=max_cpu_cores, gpu_memory_fraction=max_gpu_mem)
            with tf.device('/gpu:0'):
                timeSeries = pd.read_csv(rawDataPath)

                input_full = np.array([timeSeries['m1CMD'],
                                       timeSeries['m2CMD'],
                                       timeSeries['m3CMD'],
                                       timeSeries['m4CMD']])
                                       # timeSeries['asp'],
                                       # timeSeries['asr']])

                output_full = np.array([timeSeries['pitch'],
                                        timeSeries['roll']])

                num_samples = np.shape(input_full)
                total_train_iterations = int(np.ceil(num_samples[1] / train_data_len))
                # total_train_iterations = int(1)

                # Create the basic model to train with iteratively
                model = TFModels.drone_lstm_model(dim_in=cfg.input_size,
                                                  dim_out=cfg.output_size,
                                                  past_depth=cfg.input_depth,
                                                  layer_neurons=cfg.neurons_per_layer,
                                                  layer_dropout=cfg.layer_dropout,
                                                  learning_rate=cfg.learning_rate,
                                                  checkpoint_path='',
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
                    if (current_train_idx + train_data_len) < num_samples[1]:

                        for i in range(0, train_data_len):
                            i = current_train_idx + i

                            trainX.append(input_full[0:cfg.input_size, i:i + cfg.input_depth])
                            trainY.append(output_full[0:cfg.output_size, i + cfg.input_depth])

                    # Otherwise, only grab the remaining data we can fit
                    else:
                        for i in range(0, (num_samples[1] - current_train_idx - cfg.input_depth)):
                            i = i + current_train_idx

                            trainX.append(input_full[0:cfg.input_size, i:i + cfg.input_depth])
                            trainY.append(output_full[0:cfg.output_size, i + cfg.input_depth])

                    # Reshape for NN input (X)[batch_size, sample_depth, sample_values], (Y)[batch_size, output_values]
                    trainX = np.reshape(trainX, [-1, cfg.input_size, cfg.input_depth])
                    trainY = np.reshape(trainY, [-1, cfg.output_size])

                    current_train_idx += train_data_len

                    model.fit(trainX, trainY,
                              n_epoch=cfg.epoch_len,
                              validation_set=0.25,
                              batch_size=cfg.batch_len,
                              show_metric=True,
                              snapshot_epoch=True,
                              run_id='HereGoesNothing')

                    # ---------------------------------------
                    # Plot some data for the user to see how well training went
                    # ---------------------------------------
                    predictY = model.predict(trainX)

                    # Plot the results
                    print("Plotting Sample Outputs")
                    imgSavePath = "Checkpoints/DroneRNN_Ver3/EulerModel/Images/"

                    # PITCH
                    plt.figure(figsize=(16, 4))
                    plt.suptitle('Pitch Actual Vs Predicted')
                    plt.plot(trainY[:, 0], 'r-', label='Actual')
                    plt.plot(predictY[:, 0], 'g-', label='Predicted')
                    plt.legend()
                    plt.savefig(imgSavePath + 'pitch_iteration_' + str(train_iteration) + '.png')

                    # ROLL
                    plt.figure(figsize=(16, 4))
                    plt.suptitle('Roll Actual vs Predicted')
                    plt.plot(trainY[:, 1], 'r-', label='Actual')
                    plt.plot(predictY[:, 1], 'g-', label='Predicted')
                    plt.legend()
                    plt.savefig(imgSavePath + 'roll_iteration_' + str(train_iteration) + '.png')

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
        cfg.input_size = 4
        cfg.input_depth = max_input_depth
        cfg.output_size = 2
        cfg.batch_len = 128
        cfg.epoch_len = 10
        cfg.neurons_per_layer = 512
        cfg.layer_dropout = (0.8, 0.8)
        train_data_len = 75 * 1000  # TODO: Should this be put in the cfg object?

        # Add a few configurations for saving the model at various stages in training
        rawDataPath = 'DroneData/csv/timeSeriesDataSmoothed.csv'
        ckpt_path = 'Checkpoints/DroneRNN_Ver3/GyroModel/' + cfg.model_name + '.ckpt'
        best_ckpt_path = 'Checkpoints/DroneRNN_Ver3/GyroModel/BestResult/' + cfg.model_name + '.ckpt'
        last_ckpt_path = 'Checkpoints/DroneRNN_Ver3/GyroModel/LastResult/' + cfg.model_name + '.ckpt'

        if train_small_for_testing:
            cfg.epoch_len = 1
            cfg.neurons_per_layer = 32
            train_data_len = 10 * 1000
            last_ckpt_path = 'Checkpoints/DroneRNN_Ver3/GyroModel/SmallTesting/' + cfg.model_name + '.ckpt'

        cfg.save('gyro_model_cfg.csv')

        # Now actually do the training
        print("STARTING TRAINING OF GYRO MODEL")
        tf.reset_default_graph()
        with tf.variable_scope('gyro'):
            tflearn.init_graph(num_cores=max_cpu_cores, gpu_memory_fraction=max_gpu_mem)
            with tf.device('/gpu:0'):
                timeSeries = pd.read_csv(rawDataPath)

                input_full = np.array([timeSeries['m1CMD'],
                                       timeSeries['m2CMD'],
                                       timeSeries['m3CMD'],
                                       timeSeries['m4CMD']])
                                       # timeSeries['asp'],
                                       # timeSeries['asr']

                output_full = np.array([timeSeries['gx'],
                                        timeSeries['gy']])

                num_samples = np.shape(input_full)
                total_train_iterations = int(np.ceil(num_samples[1] / train_data_len))
                # total_train_iterations = int(1)

                model = TFModels.drone_lstm_model(dim_in=cfg.input_size,
                                                  dim_out=cfg.output_size,
                                                  past_depth=cfg.input_depth,
                                                  layer_neurons=cfg.neurons_per_layer,
                                                  layer_dropout=cfg.layer_dropout,
                                                  learning_rate=cfg.learning_rate,
                                                  checkpoint_path='',
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
                    if (current_train_idx + train_data_len) < num_samples[1]:

                        for i in range(0, train_data_len):
                            i = current_train_idx + i

                            trainX.append(input_full[0:cfg.input_size, i:i + cfg.input_depth])
                            trainY.append(output_full[0:cfg.output_size, i + cfg.input_depth])

                    # Otherwise, only grab the remaining data we can fit
                    else:
                        for i in range(0, (num_samples[1] - current_train_idx - cfg.input_depth)):
                            i = i + current_train_idx

                            trainX.append(input_full[0:cfg.input_size, i:i + cfg.input_depth])
                            trainY.append(output_full[0:cfg.output_size, i + cfg.input_depth])

                    # Reshape for NN input (X)[batch_size, sample_depth, sample_values], (Y)[batch_size, output_values]
                    trainX = np.reshape(trainX, [-1, cfg.input_size, cfg.input_depth])
                    trainY = np.reshape(trainY, [-1, cfg.output_size])

                    current_train_idx += train_data_len

                    model.fit(trainX, trainY,
                              n_epoch=cfg.epoch_len,
                              validation_set=0.25,
                              batch_size=cfg.batch_len,
                              show_metric=True,
                              snapshot_epoch=True,
                              run_id='Take2')

                    # ---------------------------------------
                    # Plot some data for the user to see how well training went
                    # ---------------------------------------
                    predictY = model.predict(trainX)

                    # # Plot the results
                    print("Plotting Sample Outputs")
                    imgSavePath = "Checkpoints/DroneRNN_Ver3/GyroModel/Images/"

                    # GYRO X
                    plt.figure(figsize=(16, 4))
                    plt.suptitle('GX Actual vs Predicted')
                    plt.plot(trainY[:, 0], 'r-', label='Actual')
                    plt.plot(predictY[:, 0], 'g-', label='Predicted')
                    plt.legend()
                    plt.savefig(imgSavePath + 'gx_iteration_' + str(train_iteration) + '.png')

                    # GYRO Y
                    plt.figure(figsize=(16, 4))
                    plt.suptitle('GY Actual vs Predicted')
                    plt.plot(trainY[:, 1], 'r-', label='Actual')
                    plt.plot(predictY[:, 1], 'g-', label='Predicted')
                    plt.legend()
                    plt.savefig(imgSavePath + 'gy_iteration_' + str(train_iteration) + '.png')

                # Save the last state of the model to a known location with a known name for quick
                # lookup in the inferencing script. This may or many not be the best performer.
                model.save(last_ckpt_path)


