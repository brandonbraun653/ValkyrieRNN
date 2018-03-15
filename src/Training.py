from __future__ import division, print_function, absolute_import

from shutil import copyfile
from src.TensorFlowModels import ModelConfig
from src.Miscellaneous import bcolors

import os
import glob
import tflearn
import numpy as np
import pandas as pd
import tensorflow as tf
import src.TensorFlowModels as TFModels

import matplotlib
matplotlib.use('Agg')   # use('Agg') for saving to file and use('TkAgg') for interactive plot
import matplotlib.pyplot as plt


class ModelTrainer:
    def __init__(self, config_path=None,  training_data_path=None, validation_data_path=None,
                 prediction_data_path=None, results_path=None):
        # --------------------------------------
        # Objects
        # --------------------------------------
        self.cfg = ModelConfig()
        self.cfg.load(config_path)

        # --------------------------------------
        # Directories
        # --------------------------------------
        self._input_config_path = config_path
        self._results_root_path = results_path + self.cfg.model_name + '/'
        self._output_config_path = self._results_root_path + 'config.csv'
        self._log_directory = self._results_root_path + 'logs/'

        # Output Training Directories
        self._training_results_path = self._results_root_path     + 'training/'
        self._epoch_checkpoint_path = self._training_results_path + 'epoch_results/'
        self._best_checkpoint_path  = self._training_results_path + 'best_results/'
        self._last_checkpoint_path  = self._training_results_path + 'last_results/'
        self._training_images_path  = self._training_results_path + 'images/'

        # Output Validation Directories
        self._validation_results_path = self._results_root_path + 'validation/'

        # Output Prediction Directories
        self._prediction_results_path = self._results_root_path + 'prediction/'

        # Input Directories
        self._train_data_path = training_data_path
        self._validation_data_path = validation_data_path
        self._prediction_data_path = prediction_data_path

        # --------------------------------------
        # Normal Variables
        # --------------------------------------
        self.validation_accuracy = []
        self.training_accuracy = []

        self._model_files = {}

        # --------------------------------------
        # Execute initialization functions
        # --------------------------------------
        # NOTE: DO NOT CHANGE THIS ORDER
        self._init_directories()
        self._init_model_data()

    def test_func(self):

        in_keys = ['m1CMD', 'm2CMD', 'm3CMD', 'm4CMD']
        out_keys = ['pitch', 'roll']

        inputs, targets, num_samples = self._parse_input_data(self._model_files['training'][0], in_keys, out_keys)

    def train_from_scratch(self, input_data_keys=None, output_data_keys=None,
                           training_plot_callback=None, validation_plot_callback=None):
        # Sanity check that our input/output size do indeed match our config file
        assert(len(input_data_keys) == self.cfg.input_size)
        assert(len(output_data_keys) == self.cfg.output_size)

        # Save so we know what data was used in the I/O modeling
        self.cfg.input_keys = input_data_keys
        self.cfg.output_keys = output_data_keys
        self.cfg.save(self._output_config_path)

        print(bcolors.OKBLUE + 'STARTING TRAINING OF: ' + self.cfg.model_name + bcolors.ENDC)
        tf.reset_default_graph()
        assert (isinstance(self.cfg.variable_scope, str))
        with tf.variable_scope(self.cfg.variable_scope):
            assert (np.isscalar(self.cfg.max_cpu_cores))
            assert (np.isscalar(self.cfg.max_gpu_mem))
            assert (isinstance(self.cfg.training_device, str))

            tflearn.init_graph(num_cores=self.cfg.max_cpu_cores, gpu_memory_fraction=self.cfg.max_gpu_mem)
            with tf.device(self.cfg.training_device):

                model = self._generate_training_model()

                # ---------------------------------------
                # Train on each file present in training folder
                # ---------------------------------------
                file_iteration = 0
                for training_file in self._model_files['training']:
                    print(bcolors.OKGREEN + 'TRAINING WITH FILE: ' + training_file + bcolors.ENDC)

                    # Grab the full set of input data
                    inputs, targets, num_samples = self._parse_input_data(filename=training_file,
                                                                          input_keys=input_data_keys,
                                                                          output_keys=output_data_keys)

                    # Split the data into parts for training so that we don't exhaust RAM resources
                    # All the data is pre-generated to save training time
                    total_train_iterations = int(np.ceil(num_samples / self.cfg.train_data_len))
                    t_inputs, t_targets = self._generate_training_data(input_full=inputs,
                                                                       output_full=targets,
                                                                       num_iter=total_train_iterations,
                                                                       total_samples=num_samples)

                    # Pre-load random validation data to reduce runtime
                    file_num = np.random.randint(len(self._model_files['validation']))
                    v_inputs, v_targets = self._generate_validation_data(self._model_files['validation'][file_num])

                    # ---------------------------------------
                    # Fit on each data set
                    # ---------------------------------------
                    for train_iteration in range(0, total_train_iterations):
                        model.fit(X_inputs=t_inputs[train_iteration],
                                  Y_targets=t_targets[train_iteration],
                                  n_epoch=self.cfg.epoch_len,
                                  batch_size=self.cfg.batch_len,
                                  validation_batch_size=self.cfg.batch_len,
                                  validation_set=(v_inputs, v_targets),
                                  show_metric=True,
                                  snapshot_epoch=True,
                                  run_id=self.cfg.model_name)

                        # Generate validation scores for training round
                        self.validation_accuracy.append(model.evaluate(X=v_inputs,
                                                                       Y=v_targets,
                                                                       batch_size=self.cfg.batch_len)[0])

                        self.training_accuracy.append(model.evaluate(X=t_inputs[train_iteration],
                                                                     Y=t_targets[train_iteration],
                                                                     batch_size=self.cfg.batch_len)[0])

                        # Generate prediction data
                        t_predict = model.predict(t_inputs[train_iteration])
                        v_predict = model.predict(v_inputs)

                        # Plot training results
                        if callable(training_plot_callback):
                            training_plot_callback(t_targets[train_iteration], t_predict)

                            img_name = 'f' + str(file_iteration) + '_iter' + str(train_iteration) + '_train.pdf'
                            plt.savefig(self._training_images_path + img_name, format='pdf', dpi=600)

                        if callable(validation_plot_callback):
                            validation_plot_callback(v_targets, v_predict)

                            img_name = 'f' + str(file_iteration) + '_iter' + str(train_iteration) + '_validate.pdf'
                            plt.savefig(self._training_images_path + img_name, format='pdf', dpi=600)

                    # ---------------------------------------
                    # Post processing/updating variables
                    # ---------------------------------------
                    file_iteration += 1

                # ---------------------------------------
                # Print validation accuracies
                # ---------------------------------------
                plt.figure(figsize=(16, 4))
                plt.suptitle('Training vs Validation Accuracy')
                plt.plot(np.array(self.validation_accuracy), 'b-', label='Validation')
                plt.plot(np.array(self.training_accuracy), 'g-', label='Training')
                plt.legend()
                plt.savefig(self._training_images_path + 'accuracyPlot.png')

    def _init_directories(self):
        """
        If a directory doesn't exist, create it. Otherwise, empty it of all files (not folders)
        :return:
        """
        def init_dir(path, clean=True):
            if not os.path.exists(path):
                os.makedirs(path)
            elif clean:
                files = glob.glob(path + '*')
                for f in files:
                    if not os.path.isdir(f):
                        os.remove(f)

        init_dir(self._results_root_path)
        init_dir(self._training_results_path)
        init_dir(self._validation_results_path)
        init_dir(self._prediction_results_path)
        init_dir(self._epoch_checkpoint_path)
        init_dir(self._best_checkpoint_path)
        init_dir(self._last_checkpoint_path)
        init_dir(self._training_images_path)
        init_dir(self._log_directory + self.cfg.model_name + '/')   # Tensorboard logs like this for some reason

    def _init_model_data(self):
        # Grab all the files available in the training, validation, and prediction paths
        self._model_files['training']   = glob.glob(self._train_data_path + '*.csv')
        self._model_files['validation'] = glob.glob(self._validation_data_path + '*.csv')
        self._model_files['prediction'] = glob.glob(self._prediction_data_path + '*.csv')

        # Copy over the configuration file for later use
        copyfile(self._input_config_path, self._output_config_path)

        # Update the model logging paths, overwriting whatever the user had
        self.cfg.load(self._output_config_path)

        self.cfg.epoch_chkpt_path = self._epoch_checkpoint_path + self.cfg.model_name
        self.cfg.best_chkpt_path = self._best_checkpoint_path + self.cfg.model_name
        self.cfg.last_chkpt_path = self._last_checkpoint_path + self.cfg.model_name
        self.cfg.image_data_path = self._training_images_path + self.cfg.model_name

        self.cfg.save(self._output_config_path)

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
        if self.cfg.model_type in TFModels.function_dispatcher:
            model_func = TFModels.function_dispatcher[self.cfg.model_type]
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
                          best_checkpoint_path=self.cfg.best_chkpt_path,
                          log_dir=self._log_directory)

    def _generate_training_data(self, input_full, output_full, num_iter, total_samples):
        output_x = [[] for i in range(num_iter)]
        output_y = [[] for i in range(num_iter)]
        current_train_idx = 0

        for iteration in range(0, num_iter):
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
            output_x[iteration], output_y[iteration] = self._reshape_data(train_x, train_y)

            current_train_idx += self.cfg.train_data_len

        return output_x, output_y

    def _generate_validation_data(self, filename):
        try:
            input_full, output_full, num_samples = self._parse_input_data(filename=filename,
                                                                          input_keys=self.cfg.input_keys,
                                                                          output_keys=self.cfg.output_keys)
        except:
            print(bcolors.WARNING + 'Validation file doesn\'t exist!' + bcolors.ENDC)
            return None, None

        validation_x, validation_y = self._fill_data(input_full, output_full, 0, num_samples)
        validation_x, validation_y = self._reshape_data(validation_x, validation_y)

        return validation_x, validation_y

    def _parse_input_data(self, filename, input_keys, output_keys):
        # --------------------------------------
        # Attempt to read all the input data requested
        # --------------------------------------
        try:
            data_set = pd.read_csv(filename)
        except:
            raise ValueError(bcolors.FAIL + 'Could not open file: ' + filename + bcolors.ENDC)

        try:
            data = []
            for key in input_keys:
                value = data_set[key]
                data.append(value)

            input_full = np.array(data)
        except:
            raise ValueError(bcolors.FAIL + 'Key does not exist in input dataset!' + bcolors.ENDC)

        try:
            data = []
            for key in output_keys:
                value = data_set[key]
                data.append(value)

            output_full = np.array(data)
        except:
            raise ValueError(bcolors.FAIL + 'Key does not exist in dataset!' + bcolors.ENDC)

        # --------------------------------------
        # Reformat for later processing
        # --------------------------------------
        # Ensures data is in row-wise format [vars x samples]
        if self.cfg.data_inversion:
            num_samples = np.shape(input_full)[1]

        # Ensures data is in column-wise format [samples x vars]
        else:
            input_full = input_full.transpose()
            output_full = output_full.transpose()
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

        # Data should be formatted row-wise [vars x samples]
        if self.cfg.data_inversion:
            for i in range(start_idx, end_idx-self.cfg.input_depth):
                inputs.append(raw_inputs[0:self.cfg.input_size, i:i + self.cfg.input_depth])
                targets.append(raw_targets[0:self.cfg.output_size, i + self.cfg.input_depth])

        # Data should be formatted column-wise [samples x vars]
        else:
            for i in range(start_idx, end_idx-self.cfg.input_depth):
                inputs.append(raw_inputs[i:i + self.cfg.input_depth, 0:self.cfg.input_size])
                targets.append(raw_targets[i + self.cfg.input_depth, 0:self.cfg.output_size])

        return inputs, targets
