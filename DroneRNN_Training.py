from __future__ import division, print_function, absolute_import

import src.TensorFlowModels as TFModels
import tflearn
import tensorflow as tf
from src.Training import ModelTrainer
from src.Miscellaneous import bcolors
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')   # use('Agg') for saving to file and use('TkAgg') for interactive plot
import matplotlib.pyplot as plt


def train_plot_callback(train_act, train_pred):
    lw = 0.8
    plt.figure(figsize=(32, 18))
    plt.suptitle('Training Data Predictions')

    plt.subplot(2, 1, 1)
    plt.plot(train_act[:, 0], 'g--', label='Pitch Actual', linewidth=lw)
    plt.plot(train_pred[:, 0], 'r-', label='Pitch Model', linewidth=2 * lw)

    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(train_act[:, 1], 'b--', label='Roll Actual', linewidth=lw)
    plt.plot(train_pred[:, 1], 'm-', label='Roll Model', linewidth=2 * lw)

    plt.legend()


def validate_plot_callback(valid_act, valid_pred):
    lw = 0.8
    plt.figure(figsize=(32, 18))
    plt.suptitle('Validation Data Predictions')

    plt.subplot(2, 1, 1)
    plt.plot(valid_act[:, 0], 'g--', label='Pitch Actual', linewidth=lw)
    plt.plot(valid_pred[:, 0], 'r-', label='Pitch Model', linewidth=2*lw)
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(valid_act[:, 1], 'b--', label='Roll Actual', linewidth=lw)
    plt.plot(valid_pred[:, 1], 'm-', label='Roll Model', linewidth=2*lw)
    plt.legend()


if __name__ == "__main__":
    train_small_for_testing     = False
    invert_data_formatting      = True
    train_euler_model           = False
    train_gyro_model            = False

    max_cpu_cores = 16
    max_gpu_mem = 0.8
    max_input_depth = 1250

    if train_small_for_testing:
        max_input_depth = 100

    # Global training paths
    big_data_path = 'G:/Projects/ValkyrieRNN/'
    small_data_path = 'C:/git/GitHub/ValkyrieRNN/'
    raw_data_path = 'G:/Projects/ValkyrieRNN/TrainingData/'

    config_path = 'C:/git/GitHub/ValkyrieRNN/euler_model_cfg.csv'
    train_data_path = 'G:/Projects/ValkyrieRNN/Data/TrainingData/'
    validation_data_path = 'G:/Projects/ValkyrieRNN/Data/ValidationData/'
    prediction_data_path = 'G:/Projects/ValkyrieRNN/Data/PredictionData/'

    results_root = 'G:/Projects/ValkyrieRNN/Simulation/'

    # Generate the model configuration for a large training set
    cfg = TFModels.ModelConfig()
    cfg.model_name = 'euler_model'
    cfg.model_type = 'drone_lstm_model_deep'
    cfg.input_size = 4
    cfg.input_depth = 250
    cfg.output_size = 2
    cfg.batch_len = 128
    cfg.epoch_len = 5
    cfg.neurons_per_layer = 128
    cfg.learning_rate = 0.002
    cfg.layer_dropout = (0.6, 0.6)
    cfg.train_data_len = 25 * 1000
    cfg.data_inversion = True

    cfg.max_cpu_cores = 16
    cfg.max_gpu_mem = 0.6
    cfg.variable_scope = 'euler'
    cfg.training_device = '/gpu:0'

    cfg.save(config_path)

    temp = ModelTrainer(config_path=config_path,
                        training_data_path=train_data_path,
                        validation_data_path=validation_data_path,
                        prediction_data_path=prediction_data_path,
                        results_path=results_root)

    in_keys = ['m1CMD', 'm2CMD', 'm3CMD', 'm4CMD']
    out_keys = ['pitch', 'roll']

    temp.train_from_scratch(input_data_keys=in_keys, output_data_keys=out_keys,
                            training_plot_callback=train_plot_callback,
                            validation_plot_callback=validate_plot_callback)

    # ---------------------------------------
    # Train the Euler Prediction Network
    # ---------------------------------------
    if train_euler_model:
        # Generate the model configuration for a large training set
        cfg = TFModels.ModelConfig()
        cfg.model_name = 'euler_model.tfl'
        cfg.model_type = 'lstm_deeply_connected'
        cfg.input_size = 4
        cfg.input_depth = max_input_depth
        cfg.output_size = 2
        cfg.batch_len = 128
        cfg.epoch_len = 10
        cfg.neurons_per_layer = 512
        cfg.learning_rate = 0.002
        cfg.layer_dropout = (0.5, 0.5)
        cfg.train_data_len = 75 * 1000
        cfg.epoch_chkpt_path = big_data_path + 'Checkpoints/EulerModel/EpochResults/' + cfg.model_name + '.ckpt'
        cfg.best_chkpt_path = big_data_path + 'Checkpoints/EulerModel/BestResults/' + cfg.model_name + '.ckpt'
        cfg.last_chkpt_path = big_data_path + 'Checkpoints/EulerModel/LastResult/' + cfg.model_name + '.ckpt'
        cfg.image_data_path = big_data_path + 'Checkpoints/EulerModel/Images/'

        if train_small_for_testing:
            cfg.epoch_len = 1
            cfg.neurons_per_layer = 32
            cfg.train_data_len = 10 * 1000
            cfg.epoch_chkpt_path = 'Checkpoints/DroneRNN_Ver3/EulerModel/' + cfg.model_name + '.ckpt'
            cfg.best_chkpt_path = 'Checkpoints/DroneRNN_Ver3/EulerModel/BestResult/' + cfg.model_name + '.ckpt'
            cfg.last_chkpt_path = 'Checkpoints/DroneRNN_Ver3/EulerModel/SmallTesting/' + cfg.model_name + '.ckpt'
            cfg.image_data_path = 'Checkpoints/DroneRNN_Ver3/EulerModel/Images/'

        if invert_data_formatting:
            cfg.data_inversion = True
        else:
            cfg.data_inversion = False

        cfg.save('euler_model_cfg.csv')

        # Now actually do the training
        print(bcolors.OKBLUE + "STARTING TRAINING OF EULER MODEL" + bcolors.ENDC)
        tf.reset_default_graph()
        with tf.variable_scope('euler'):
            tflearn.init_graph(num_cores=max_cpu_cores, gpu_memory_fraction=max_gpu_mem)
            with tf.device('/gpu:0'):

                train_gen = DataServer(path=raw_data_path, name_pattern='timeSeriesDataNoisy', file_ending='.csv')

                if invert_data_formatting:
                    model_input_shape = [None, cfg.input_size, cfg.input_depth]
                else:
                    model_input_shape = [None, cfg.input_depth, cfg.input_size]

                # Create the basic model to train with iteratively
                model = TFModels.drone_lstm_deeply_connected(shape=model_input_shape,
                                                       dim_in=cfg.input_size,
                                                       dim_out=cfg.output_size,
                                                       past_depth=cfg.input_depth,
                                                       layer_neurons=cfg.neurons_per_layer,
                                                       layer_dropout=cfg.layer_dropout,
                                                       learning_rate=cfg.learning_rate,
                                                       checkpoint_path=cfg.epoch_chkpt_path,
                                                       best_checkpoint_path=cfg.best_chkpt_path)

                file_count = -1

                while True:
                    file_count += 1
                    timeSeries = train_gen.get_next_dataset()
                    if timeSeries is None:
                        break

                    print(bcolors.OKGREEN + "TRAINING WITH FILE: ", train_gen.current_file_name + bcolors.ENDC)
                    if invert_data_formatting:
                        input_full = np.array([timeSeries['m1CMD'],
                                               timeSeries['m2CMD'],
                                               timeSeries['m3CMD'],
                                               timeSeries['m4CMD']])

                        output_full = np.array([timeSeries['pitch'],
                                                timeSeries['roll']])

                        num_samples = np.shape(input_full)[1]

                    else:
                        input_full = np.array([timeSeries['m1CMD'],
                                               timeSeries['m2CMD'],
                                               timeSeries['m3CMD'],
                                               timeSeries['m4CMD']]).transpose()

                        output_full = np.array([timeSeries['pitch'],
                                                timeSeries['roll']]).transpose()

                        num_samples = np.shape(input_full)[0]

                    # -------------------------------------------
                    # Train with the full dataset being broken up into
                    # multiple parts to handle RAM overload
                    # -------------------------------------------
                    current_train_idx = 0
                    trainX = []
                    trainY = []
                    total_train_iterations = int(np.ceil(num_samples / cfg.train_data_len))
                    for train_iteration in range(0, total_train_iterations):
                        # Reset the training data for a new batch
                        trainX = []
                        trainY = []

                        # Grab a full set of data if we have enough left
                        if (current_train_idx + cfg.train_data_len) < num_samples:

                            for i in range(0, cfg.train_data_len):
                                i = current_train_idx + i

                                if invert_data_formatting:
                                    trainX.append(input_full[0:cfg.input_size, i:i + cfg.input_depth])
                                    trainY.append(output_full[0:cfg.output_size, i + cfg.input_depth])

                                else:
                                    trainX.append(input_full[i:i + cfg.input_depth, 0:cfg.input_size])
                                    trainY.append(output_full[i + cfg.input_depth, 0:cfg.output_size])

                        # Otherwise, only grab the remaining data we can fit
                        else:
                            for i in range(0, (num_samples - current_train_idx - cfg.input_depth)):
                                i = i + current_train_idx

                                if invert_data_formatting:
                                    trainX.append(input_full[0:cfg.input_size, i:i + cfg.input_depth])
                                    trainY.append(output_full[0:cfg.output_size, i + cfg.input_depth])

                                else:
                                    trainX.append(input_full[i:i + cfg.input_depth, 0:cfg.input_size])
                                    trainY.append(output_full[i + cfg.input_depth, 0:cfg.output_size])

                        # Reshape for NN input
                        if invert_data_formatting:
                            trainX = np.reshape(trainX, [-1, cfg.input_size, cfg.input_depth])
                            trainY = np.reshape(trainY, [-1, cfg.output_size])

                        else:
                            trainX = np.reshape(trainX, [-1, cfg.input_depth, cfg.input_size])
                            trainY = np.reshape(trainY, [-1, cfg.output_size])

                        current_train_idx += cfg.train_data_len

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
                        print(bcolors.OKGREEN + "Plotting Sample Outputs" + bcolors.ENDC)
                        # PITCH
                        plt.figure(figsize=(16, 4))
                        plt.suptitle('Pitch Actual Vs Predicted')
                        plt.plot(trainY[:, 0], 'r-', label='Actual')
                        plt.plot(predictY[:, 0], 'g-', label='Predicted')
                        plt.legend()
                        plt.savefig(cfg.image_data_path + 'dataset_' + str(file_count) +
                                    '_pitch_iteration_' + str(train_iteration) + '.png')

                        # ROLL
                        plt.figure(figsize=(16, 4))
                        plt.suptitle('Roll Actual vs Predicted')
                        plt.plot(trainY[:, 1], 'r-', label='Actual')
                        plt.plot(predictY[:, 1], 'g-', label='Predicted')
                        plt.legend()
                        plt.savefig(cfg.image_data_path + 'dataset_' + str(file_count) +
                                    '_roll_iteration_' + str(train_iteration) + '.png')

                    # Save the model state more as a sanity check than anything else
                    model.save(cfg.last_chkpt_path)

    # ---------------------------------------
    # Train the Gyro Prediction Network
    # ---------------------------------------
    if train_gyro_model:
        # Generate the model configuration for a large training set
        cfg = TFModels.ModelConfig()
        cfg.model_name = 'gyro_model.tfl'
        cfg.model_type = 'lstm_deeply_connected'
        cfg.input_size = 4
        cfg.input_depth = max_input_depth
        cfg.output_size = 2
        cfg.batch_len = 128
        cfg.epoch_len = 2
        cfg.neurons_per_layer = 32
        cfg.learning_rate = 0.002
        cfg.layer_dropout = (0.3, 0.3)
        cfg.train_data_len = 75 * 1000
        cfg.epoch_chkpt_path = big_data_path + 'Checkpoints/GyroModel/EpochResults/' + cfg.model_name + '.ckpt'
        cfg.best_chkpt_path  = big_data_path + 'Checkpoints/GyroModel/BestResults/' + cfg.model_name + '.ckpt'
        cfg.last_chkpt_path  = big_data_path + 'Checkpoints/GyroModel/LastResult/' + cfg.model_name + '.ckpt'
        cfg.image_data_path  = big_data_path + 'Checkpoints/GyroModel/Images/'

        if train_small_for_testing:
            cfg.epoch_len = 1
            cfg.neurons_per_layer = 32
            cfg.train_data_len = 10 * 1000
            cfg.epoch_chkpt_path = 'Checkpoints/DroneRNN_Ver3/GyroModel/' + cfg.model_name + '.ckpt'
            cfg.best_chkpt_path  = 'Checkpoints/DroneRNN_Ver3/GyroModel/BestResult/' + cfg.model_name + '.ckpt'
            cfg.last_chkpt_path  = 'Checkpoints/DroneRNN_Ver3/GyroModel/SmallTesting/' + cfg.model_name + '.ckpt'
            cfg.image_data_path  = 'Checkpoints/DroneRNN_Ver3/GyroModel/Images/'

        if invert_data_formatting:
            cfg.data_inversion = True
        else:
            cfg.data_inversion = False

        cfg.save('gyro_model_cfg.csv')

        # Now actually do the training
        print(bcolors.OKBLUE + 'STARTING TRAINING OF GYRO MODEL' + bcolors.ENDC)
        tf.reset_default_graph()
        with tf.variable_scope('gyro'):

            tflearn.init_graph(num_cores=max_cpu_cores, gpu_memory_fraction=max_gpu_mem)

            with tf.device('/gpu:0'):
                train_gen = DataServer(path=raw_data_path, name_pattern='timeSeriesDataNoisy', file_ending='.csv')

                if invert_data_formatting:
                    model_input_shape = [None, cfg.input_size, cfg.input_depth]
                else:
                    model_input_shape = [None, cfg.input_depth, cfg.input_size]

                # Create the basic model to train with iteratively
                model = TFModels.drone_lstm_deeply_connected(shape=model_input_shape,
                                                       dim_in=cfg.input_size,
                                                       dim_out=cfg.output_size,
                                                       past_depth=cfg.input_depth,
                                                       layer_neurons=cfg.neurons_per_layer,
                                                       layer_dropout=cfg.layer_dropout,
                                                       learning_rate=cfg.learning_rate,
                                                       checkpoint_path=cfg.epoch_chkpt_path,
                                                       best_checkpoint_path=cfg.best_chkpt_path)

                file_count = -1

                while True:
                    file_count += 1
                    timeSeries = train_gen.get_next_dataset()
                    if timeSeries is None:
                        break

                    print(bcolors.OKGREEN + "TRAINING WITH FILE: ", train_gen.current_file_name + bcolors.ENDC)
                    if invert_data_formatting:
                        input_full = np.array([timeSeries['m1CMD'],
                                               timeSeries['m2CMD'],
                                               timeSeries['m3CMD'],
                                               timeSeries['m4CMD']])

                        output_full = np.array([timeSeries['gx'],
                                                timeSeries['gy']])

                        num_samples = np.shape(input_full)[1]

                    else:
                        input_full = np.array([timeSeries['m1CMD'],
                                               timeSeries['m2CMD'],
                                               timeSeries['m3CMD'],
                                               timeSeries['m4CMD']]).transpose()

                        output_full = np.array([timeSeries['gx'],
                                                timeSeries['gy']]).transpose()

                        num_samples = np.shape(input_full)[0]

                    # -------------------------------------------
                    # Train with the full dataset being broken up into
                    # multiple parts to handle RAM overload
                    # -------------------------------------------
                    current_train_idx = 0
                    trainX = []
                    trainY = []
                    total_train_iterations = int(np.ceil(num_samples / cfg.train_data_len))
                    for train_iteration in range(0, total_train_iterations):
                        # Reset the training data for a new batch
                        trainX = []
                        trainY = []

                        # Grab a full set of data if we have enough left
                        if (current_train_idx + cfg.train_data_len) < num_samples:

                            for i in range(0, cfg.train_data_len):
                                i = current_train_idx + i

                                if invert_data_formatting:
                                    trainX.append(input_full[0:cfg.input_size, i:i + cfg.input_depth])
                                    trainY.append(output_full[0:cfg.output_size, i + cfg.input_depth])

                                else:
                                    trainX.append(input_full[i:i + cfg.input_depth, 0:cfg.input_size])
                                    trainY.append(output_full[i + cfg.input_depth, 0:cfg.output_size])

                        # Otherwise, only grab the remaining data we can fit
                        else:
                            for i in range(0, (num_samples - current_train_idx - cfg.input_depth)):
                                i = i + current_train_idx

                                if invert_data_formatting:
                                    trainX.append(input_full[0:cfg.input_size, i:i + cfg.input_depth])
                                    trainY.append(output_full[0:cfg.output_size, i + cfg.input_depth])

                                else:
                                    trainX.append(input_full[i:i + cfg.input_depth, 0:cfg.input_size])
                                    trainY.append(output_full[i + cfg.input_depth, 0:cfg.output_size])

                        # Reshape for NN input
                        if invert_data_formatting:
                            trainX = np.reshape(trainX, [-1, cfg.input_size, cfg.input_depth])
                            trainY = np.reshape(trainY, [-1, cfg.output_size])

                        else:
                            trainX = np.reshape(trainX, [-1, cfg.input_depth, cfg.input_size])
                            trainY = np.reshape(trainY, [-1, cfg.output_size])

                        current_train_idx += cfg.train_data_len

                        model.fit(trainX, trainY,
                                  n_epoch=cfg.epoch_len,
                                  validation_set=0.25,
                                  batch_size=cfg.batch_len,
                                  show_metric=True,
                                  snapshot_epoch=True,
                                  run_id='GyroMotion')

                        # ---------------------------------------
                        # Plot some data for the user to see how well training went
                        # ---------------------------------------
                        predictY = model.predict(trainX)

                        # Plot the results
                        print(bcolors.OKGREEN + "Plotting Sample Outputs" + bcolors.ENDC)

                        # GYRO X
                        plt.figure(figsize=(16, 4))
                        plt.suptitle('GX Actual vs Predicted')
                        plt.plot(trainY[:, 0], 'r-', label='Actual')
                        plt.plot(predictY[:, 0], 'g-', label='Predicted')
                        plt.legend()
                        plt.savefig(cfg.image_data_path + 'dataset_' + str(file_count) +
                                    '_gx_iteration_' + str(train_iteration) + '.png')

                        # GYRO Y
                        plt.figure(figsize=(16, 4))
                        plt.suptitle('GY Actual vs Predicted')
                        plt.plot(trainY[:, 1], 'r-', label='Actual')
                        plt.plot(predictY[:, 1], 'g-', label='Predicted')
                        plt.legend()
                        plt.savefig(cfg.image_data_path + 'dataset_' + str(file_count) +
                                    '_gy_iteration_' + str(train_iteration) + '.png')

                # Save the last state of the model to a known location with a known name for quick
                # lookup in the inferencing script. This may or many not be the best performer.
                model.save(cfg.last_chkpt_path)


