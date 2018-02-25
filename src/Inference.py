from __future__ import division, print_function, absolute_import

import tflearn
import tensorflow
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use('TkAgg')
import matplotlib.pyplot as graph


import src.TensorFlowModels as Models
import src.MotorController
from src.ControlSystem import AxisController

"""
IMPORTANT NOTES:

1. Process all the raw data from the simulation into a Matlab file that will smooth and then calculate the step response
characteristics. See:
     https://www.mathworks.com/help/control/ref/stepinfo.html
     https://www.mathworks.com/help/matlab/ref/smoothdata.html
     https://www.mathworks.com/help/matlab/matlab_external/call-user-script-and-function-from-python.html
     https://www.mathworks.com/help/matlab/matlab_external/handle-data-returned-from-matlab-to-python.html
     
     
inputs = [np.arange(5), np.arange(5), np.arange(5), np.arange(5)]
inputs = np.reshape(inputs, (4, 5))
print(inputs)

inputs = np.roll(inputs, 1, axis=1)
print(inputs)

for i in range(5, 10):
    print(i)
    inputs[:,0] = np.array([i, i, i, i])
    print(inputs)

    print('Rolling...')
    inputs = np.roll(inputs, 1, axis=1)
    print(inputs)
    print('\n')
"""


class StepResponseMetrics:
    def __init__(self):
        self.rise_time = 0
        self.settling_time = 0
        self.percent_overshoot = 0
        self.steady_state_error = 0


class DroneModel:
    def __init__(self, tf_cell_type='rnn', tf_model_name='', tf_chkpt_path=''):
        self.model_name = tf_model_name
        self.checkpoint_path = tf_chkpt_path

        self._tf_cell_type = tf_cell_type
        self._tf_model = None

        self._model_csv_access_keys = ['dim_in', 'dim_out', 'input_depth', 'neurons', 'dropout', 'learning_rate']
        self._model_input_dim = 0
        self._model_output_dim = 0
        self._model_input_depth = 0
        self._model_layer_neurons = 0
        self._model_layer_dropout = (0, 0)
        self._model_learning_rate = 0

    def initialize(self, configuration_path):
        """
        Do things like set up the time series inputs for things like
        step functions, buffer sizes, NN model, Matlab environment etc
        :return:
        """

        assert self._tf_cell_type == 'rnn' or self._tf_cell_type == 'lstm', "Incorrect cell type input"

        if self._tf_cell_type == 'rnn':
            self._tf_model = Models.drone_rnn_model(dim_in=4, dim_out=2, past_depth=1250, layer_neurons=256,
                                                    layer_dropout=(0.8, 0.8), learning_rate=0.002,
                                                    checkpoint_path=self.checkpoint_path)

        else:
            self._tf_model = Models.drone_lstm_model(dim_in=4, dim_out=2, past_depth=1250, layer_neurons=256,
                                                     layer_dropout=(0.8, 0.8), learning_rate=0.001,
                                                     checkpoint_path=self.checkpoint_path)

        self._tf_model.load(self.checkpoint_path + self.model_name)
        print('Loaded archived model: ', self.model_name)

    def set_rate_pid(self, kp, ki, kd):
        raise NotImplementedError

    def set_angle_pid(self, kp, ki, kd):
        raise NotImplementedError

    def simulate_pitch_step(self, step_size, sim_length):
        """

        :param step_size:
        :param sim_length:
        :return: (1 x sim_length) ndarray of pitch response
        """
        raise NotImplementedError

    def simulate_roll_step(self, step_size, sim_length):
        raise NotImplementedError

    def simulate_coupled_pitch_roll_step(self, step_size_pitch, step_size_roll, sim_length):
        """
        The goal here is to allow for simulation of coupled axis movements
        :param step_size_pitch:
        :param step_size_roll:
        :param sim_length:
        :return:
        """
        raise NotImplementedError

    def __smooth_raw_output(self):
        """
        Implements a smoothing filter so that we don't have so much noise to
        process
        :return:
        """
        raise NotImplementedError

    def __parse_model_config(self):
        raise NotImplementedError


if __name__ == "__main__":
    print('Hello')
