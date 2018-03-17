import os
import glob
import numpy as np
import pandas as pd

from src.Miscellaneous import bcolors


class DataHandler:
    def __init__(self, cfg):
        self.cfg = cfg

    def generate_training_data(self, input_raw, output_raw, num_train_sets, total_samples):
        """
        Takes a set of [input, output] data files and splits it up into training sets

        :param input_raw: the full set of input data
        :param output_raw: the full set of output target data
        :param num_train_sets: how many sets to split the input data into
        :param total_samples: length of the input data
        :return:
        """
        output_x = [[] for i in range(num_train_sets)]
        output_y = [[] for i in range(num_train_sets)]
        current_train_idx = 0

        for iteration in range(0, num_train_sets):
            # Grab a full set of data if we have enough left
            if (current_train_idx + self.cfg.train_data_len) < total_samples:
                train_x, train_y = self.fill_data(raw_inputs=input_raw,
                                                  raw_targets=output_raw,
                                                  start_idx=current_train_idx,
                                                  end_idx=current_train_idx+self.cfg.train_data_len)

            # Otherwise, only get remaining data
            else:
                train_x, train_y = self.fill_data(raw_inputs=input_raw,
                                                  raw_targets=output_raw,
                                                  start_idx=current_train_idx,
                                                  end_idx=(total_samples - current_train_idx - self.cfg.input_depth))

            # Return the data in the correct format for direct input into the network
            output_x[iteration], output_y[iteration] = self.reshape_data(train_x, train_y)

            current_train_idx += self.cfg.train_data_len

        return output_x, output_y

    def generate_validation_data(self, filename):
        try:
            input_full, output_full, num_samples = self.parse_input_data(filename=filename,
                                                                         input_keys=self.cfg.input_keys,
                                                                         output_keys=self.cfg.output_keys)
        except:
            print(bcolors.WARNING + 'Validation file doesn\'t exist!' + bcolors.ENDC)
            return None, None

        validation_x, validation_y = self.fill_data(input_full, output_full, 0, num_samples)
        validation_x, validation_y = self.reshape_data(validation_x, validation_y)

        return validation_x, validation_y

    def parse_input_data(self, filename, input_keys, output_keys):
        """
        Takes an input csv file and reads out a set of data specified by the input/output keys

        :param filename: The file to read from
        :param input_keys: Input data to build up
        :param output_keys: Output data to build up
        :return: input_full, output_full, num_samples
        """
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

    def reshape_data(self, inputs, targets):
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

    def fill_data(self, raw_inputs, raw_targets, start_idx, end_idx):
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
