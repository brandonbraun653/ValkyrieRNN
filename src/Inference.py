from __future__ import division, print_function, absolute_import

import tflearn
import tensorflow as tf
import numpy as np
import pandas as pd
import matlab.engine
import matplotlib; matplotlib.use('TkAgg')
import matplotlib.pyplot as graph


import src.TensorFlowModels as TFModels
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
    def __init__(self, tf_euler_chkpt_path='', tf_gyro_chkpt_path=''):
        self._euler_chkpt_path = tf_euler_chkpt_path
        self._gyro_chkpt_path = tf_gyro_chkpt_path

        self._euler_model = None    # Predicts Euler Angle Outputs
        self._gyro_model = None     # Predicts Gyro Sensor Outputs

        self._euler_graph = tf.Graph()
        self._gyro_graph = tf.Graph()

        self._matlab_engine = None

        self._pitch_ctrl = None
        self._roll_ctrl = None
        self._yaw_ctrl = None

        self._pitch_angle_setpoint = 0
        self._pitch_angle_feedback = 0
        self._pitch_rate_feedback = 0
        self._pitch_output = 0
        self._sample_time = 4.0

    def initialize(self, euler_cfg_path, gyro_cfg_path):
        """
        Do things like set up the time series inputs for things like
        step functions, buffer sizes, NN model, Matlab environment etc
        :return:
        """

        # Setup the Euler prediction network
        # with self._euler_graph.as_default():
        #     print("Initializing Euler Model...")
        #     euler_cfg = TFModels.ModelConfig()
        #     euler_cfg.load(euler_cfg_path)
        #
        #     self._euler_model = TFModels.drone_rnn_model(dim_in=euler_cfg.input_size,
        #                                                  dim_out=euler_cfg.output_size,
        #                                                  past_depth=euler_cfg.input_depth,
        #                                                  layer_neurons=euler_cfg.neurons_per_layer,
        #                                                  layer_dropout=euler_cfg.layer_dropout,
        #                                                  learning_rate=euler_cfg.learning_rate)
        #
        #     self._euler_model.load(self._euler_chkpt_path)
        #     print("Loaded archived Euler model.")

        # Setup the Gyro prediction network
        # with self._gyro_graph.as_default():
        #     print("Initializing Gyro Model...")
        #     gyro_cfg = TFModels.ModelConfig()
        #     gyro_cfg.load(gyro_cfg_path)
        #
        #     self._gyro_model = TFModels.drone_rnn_model(dim_in=gyro_cfg.input_size,
        #                                                 dim_out=gyro_cfg.output_size,
        #                                                 past_depth=gyro_cfg.input_depth,
        #                                                 layer_neurons=gyro_cfg.neurons_per_layer,
        #                                                 layer_dropout=gyro_cfg.layer_dropout,
        #                                                 learning_rate=gyro_cfg.learning_rate)
        #
        #     self._gyro_model.load(self._gyro_chkpt_path)
        #     print("Loaded archived Gyro model.")

        # Setup the Matlab environment
        # print("Starting Matlab Engine")
        # self._matlab_engine = matlab.engine.start_matlab()
        # self._matlab_engine.addpath(r'C:\git\GitHub\ValkyrieRNN\Scripts\Matlab', nargout=0)
        # print("Done")

        # Setup the PID controllers
        self._pitch_ctrl = AxisController(angle_setpoint=self._pitch_angle_setpoint,
                                          angle_fb=self._pitch_angle_feedback,
                                          rate_fb=self._pitch_rate_feedback,
                                          output=self._pitch_output,
                                          sample_time_ms=self._sample_time)

    def TESTFUNC(self):
        self._pitch_ctrl.update_angle_pid(2.5, 3.0, 0.01)
        self._pitch_ctrl.update_rate_pid(0.9, 4.0, 0.05)

        x = np.linspace(0.0, 20*np.pi, 10*1000)

        for val in x:
            # Update PID controller inputs
            self._pitch_angle_setpoint = np.sin(val)
            self._pitch_angle_feedback = np.cos(val)
            self._pitch_rate_feedback = np.sin(val)*np.cos(val)
            print("Angle Setpoint: ", str(self._pitch_angle_setpoint))
            print("Angle Feedback: ", str(self._pitch_angle_feedback))
            print("Rate Feedback: ", str(self._pitch_rate_feedback))

            # Update the controller
            self._pitch_ctrl.compute()
            print("PID Output: ", str(self._pitch_output), "\n")






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
