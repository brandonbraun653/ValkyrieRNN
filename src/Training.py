from __future__ import division, print_function, absolute_import


import os
import tflearn
import numpy as np
import pandas as pd
import tensorflow as tf
import src.TensorFlowModels as TFModels

from src.TensorFlowModels import ModelConfig
from src.Miscellaneous import bcolors


class DataServer:
    """
    The purpose of this class is to produce training data for a model given
    an arbitrary naming convention.
    """
    def __init__(self, path='', name_pattern='', file_ending=''):
        self._file_index = 0
        self._file_path = path
        self._file_pattern = name_pattern
        self._file_ending = file_ending

        self.current_file_name = None

    def reset_location(self):
        self._file_index = 0

    def get_next_dataset(self):
        file = self._get_next_file()
        self.current_file_name = file

        try:
            return pd.read_csv(file)

        except:
            print("Ran out of training data!")
            return None

    def _get_next_file(self):
        next_file_name = self._file_path + self._file_pattern + str(self._file_index) + self._file_ending
        self._file_index += 1

        return next_file_name


class ModelTrainer:
    def __init__(self, config_path=None,  training_data_path=None, validation_data_path=None,
                 prediction_data_path=None, data_file_naming_pattern=None, results_path=None):

        # --------------------------------------
        # Directories
        # --------------------------------------
        self._results_root_path = results_path
        self._training_results_path = results_path + 'training/'
        self._validation_results_path = results_path + 'validation/'
        self._prediction_results_path = results_path + 'prediction/'

        self._train_data_path = training_data_path
        self._validation_data_path = validation_data_path
        self._prediction_data_path = prediction_data_path
        self._data_file_naming_pattern = data_file_naming_pattern

        # --------------------------------------
        # Normal Variables
        # --------------------------------------
        self.training_loss = None
        self.training_accuracy = None

        # --------------------------------------
        # Objects
        # --------------------------------------
        self.cfg = ModelConfig()
        self.cfg.load(config_path)

        # --------------------------------------
        # Execute initialization functions
        # --------------------------------------
        self._init_directories()

    def train_from_scratch(self):
        print(bcolors.OKBLUE + 'STARTING TRAINING OF ' + self.cfg.model_name + bcolors.ENDC)
        tf.reset_default_graph()

        assert (isinstance(self.cfg.variable_scope, str))

        # Ensure we don't get naming collisions between multiple models
        with tf.variable_scope(self.cfg.variable_scope):
            assert (isinstance(self.cfg.max_cpu_cores, int))
            assert (isinstance(self.cfg.max_gpu_mem, float))
            assert (isinstance(self.cfg.training_device, str))

            tflearn.init_graph(num_cores=self.cfg.max_cpu_cores, gpu_memory_fraction=self.cfg.max_gpu_mem)
            with tf.device(self.cfg.training_device):

                # File generator objects to load data sets on command
                train_gen = DataServer(path=self._train_data_path,
                                       name_pattern=self._data_file_naming_pattern,
                                       file_ending='.csv')

                validate_gen = DataServer(path=self._validation_data_path,
                                          name_pattern=self._data_file_naming_pattern,
                                          file_ending='.csv')

                # Model used for training
                model = self._generate_training_model()

                # Loop through all the data in the training folder
                file_count = -1
                while True:
                    file_count += 1

                    time_series = train_gen.get_next_dataset()
                    if time_series is None:
                        break

                    print(bcolors.OKGREEN + "TRAINING WITH FILE: ", train_gen.current_file_name + bcolors.ENDC)
                    inputs, targets, num_samples = self._parse_input_data(data_set=time_series)

                    # -------------------------------------------
                    # Train with the full dataset being broken up into
                    # multiple parts to handle RAM overload
                    # -------------------------------------------
                    train_data_idx = 0
                    total_train_iterations = int(np.ceil(num_samples / self.cfg.train_data_len))
                    for train_iteration in range(0, total_train_iterations):

                        # Generate the training data for this iteration
                        model_inputs, model_targets = self._generate_training_data(input_full=inputs,
                                                                                   output_full=targets,
                                                                                   current_train_idx=train_data_idx,
                                                                                   total_samples=num_samples)
                        # TODO: Can I increment this inside ^^^ function?
                        train_data_idx += self.cfg.train_data_len

                        # Train with the current data set
                        model.fit(X_inputs=model_inputs,
                                  Y_targets=model_targets,
                                  n_epoch=self.cfg.epoch_len,
                                  validation_set=0.25,
                                  batch_size=self.cfg.batch_len,
                                  show_metric=True,
                                  snapshot_epoch=True,
                                  run_id=self.cfg.model_name)

                        # ---------------------------------------
                        # Generate validation scores for training round
                        # ---------------------------------------
                        validation_file = self._validation_data_path + 'validation_set.csv'
                        v_inputs, v_targets = self._generate_validation_data(validation_file)
                        scores = model.evaluate(X=v_inputs, Y=v_targets, batch_size=self.cfg.batch_len)

                        # TODO: Do something with the scores, like logging
                        print(scores)

                        get this running first before continuing!!


    def _init_directories(self):
        # Ensure the root results directory exists
        if not os.path.exists(self._results_root_path):
            os.makedirs(self._results_root_path)

        if not os.path.exists(self._training_results_path):
            os.makedirs(self._training_results_path)

        if not os.path.exists(self._validation_results_path):
            os.makedirs(self._validation_results_path)

        if not os.path.exists(self._prediction_results_path):
            os.makedirs(self._prediction_results_path)

    def _generate_training_model(self):
        """
        A factory function to generate a system model based upon values in the
        class config object
        """
        if self.cfg.data_inversion:
            model_input_shape = [None, self.cfg.input_size, self.cfg.input_depth]
        else:
            model_input_shape = [None, self.cfg.input_depth, self.cfg.input_size]

        # Grab the particular model function in use
        assert (isinstance(self.cfg.model_type, str))

        if self.cfg.model_type == 'drone_rnn_model':
            model_func = TFModels.drone_rnn_model
        elif self.cfg.model_type == 'drone_lstm_model_deep':
            model_func = TFModels.drone_lstm_model_deep
        elif self.cfg.model_type == 'drone_lstm_deeply_connected':
            model_func = TFModels.drone_lstm_deeply_connected
        else:
            raise ValueError(bcolors.FAIL + 'Invalid model type!' + bcolors.ENDC)

        # All models are guaranteed to have this input form
        return model_func(shape=model_input_shape,
                          dim_in=self.cfg.input_size,
                          dim_out=self.cfg.output_size,
                          past_depth=self.cfg.input_depth,
                          layer_neurons=self.cfg.neurons_per_layer,
                          layer_dropout=self.cfg.layer_dropout,
                          learning_rate=self.cfg.learning_rate,
                          checkpoint_path=self.cfg.epoch_chkpt_path,
                          best_checkpoint_path=self.cfg.best_chkpt_path)

    def _generate_training_data(self, input_full, output_full, current_train_idx, total_samples):
        # Grab a full set of data if we have enough left
        if (current_train_idx + self.cfg.train_data_len) < total_samples:
            train_x, train_y = self._fill_data(raw_inputs=input_full,
                                               raw_targets=output_full,
                                               start_idx=current_train_idx,
                                               end_idx=current_train_idx+self.cfg.train_data_len)

        # Otherwise, only get remaining data
        else:
            train_x, train_y = self._fill_data(raw_inputs=input_full,
                                               raw_targets=output_full,
                                               start_idx=current_train_idx,
                                               end_idx=(total_samples - current_train_idx - self.cfg.input_depth))

        # Return the data in the correct format for direct input into the network
        return self._reshape_data(train_x, train_y)

    def _generate_validation_data(self, filename):
        try:
            input_full, output_full, num_samples = self._parse_input_data(pd.read_csv(filename))
        except:
            print("Validation file doesn't exist!")
            return None

        train_x, train_y = self._fill_data(input_full, output_full, 0, num_samples)
        train_x, train_y = self._reshape_data(train_x, train_y)

        return train_x, train_y

    def _parse_input_data(self, data_set):
        if self.cfg.data_inversion:
            # Ensures data is in row-wise format [vars x samples]
            input_full = np.array([data_set['m1CMD'],
                                   data_set['m2CMD'],
                                   data_set['m3CMD'],
                                   data_set['m4CMD']])

            output_full = np.array([data_set['pitch'],
                                    data_set['roll']])

            num_samples = np.shape(input_full)[1]

        else:
            # Ensures data is in column-wise format [samples x vars]
            input_full = np.array([data_set['m1CMD'],
                                   data_set['m2CMD'],
                                   data_set['m3CMD'],
                                   data_set['m4CMD']]).transpose()

            output_full = np.array([data_set['pitch'],
                                    data_set['roll']]).transpose()

            num_samples = np.shape(input_full)[0]

        return input_full, output_full, num_samples

    def _reshape_data(self, inputs, targets):
        """
        Reshapes the input data into the correct form for the neural network
        :param inputs:
        :param targets:
        :return:
        """

        if self.cfg.data_inversion:
            inputs = np.reshape(inputs, [-1, self.cfg.input_size, self.cfg.input_depth])
            targets = np.reshape(targets, [-1, self.cfg.output_size])

        else:
            inputs = np.reshape(inputs, [-1, self.cfg.input_depth, self.cfg.input_size])
            targets = np.reshape(targets, [-1, self.cfg.output_size])

        return inputs, targets

    def _fill_data(self, raw_inputs, raw_targets, start_idx, end_idx):
        """
        Takes a set of input/target data and creates two new data sets that are a subset
        of the original. That subset spans from [start_idx, end_idx]

        :param raw_inputs:
        :param raw_targets:
        :param start_idx:
        :param end_idx:
        :return:
        """
        inputs = []
        targets = []

        for i in range(start_idx, end_idx):
            # Data should be formatted row-wise [vars x samples]
            if self.cfg.data_inversion:
                inputs.append(raw_inputs[0:self.cfg.input_size, i:i + self.cfg.input_depth])
                targets.append(raw_targets[0:self.cfg.output_size, i + self.cfg.input_depth])

            # Data should be formatted column-wise [samples x vars]
            else:
                inputs.append(raw_inputs[i:i + self.cfg.input_depth, 0:self.cfg.input_size])
                targets.append(raw_targets[i + self.cfg.input_depth, 0:self.cfg.output_size])

        return inputs, targets
