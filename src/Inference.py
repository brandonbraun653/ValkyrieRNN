from __future__ import division, print_function, absolute_import

import tflearn
import tensorflow
import numpy as np
import matplotlib; matplotlib.use('TkAgg')
import matplotlib.pyplot as graph


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
     
2. BEFORE coding anything else, make sure the basic communication idea actually works.
Create a c++ project that gives simulation parameters to a python file and then have it immediately return some fudged
performance metrics that you need. Probably should use a string format with a ',' as delimiter and '\n' as EOT.
"""


class StepResponseMetrics:
    def __init__(self):
        self.rise_time = 0
        self.settling_time = 0
        self.percent_overshoot = 0
        self.steady_state_error = 0


class DroneModel:
    def __init__(self):
        self.name = ""

    def initialize(self):
        """
        Do things like set up the time series inputs for things like
        step functions, buffer sizes, NN model, Matlab environment etc
        :return:
        """
        raise NotImplementedError

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


if __name__ == "__main__":
    print('Hello')
