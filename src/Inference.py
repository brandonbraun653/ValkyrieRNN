from __future__ import division, print_function, absolute_import

import time
import tflearn
import tensorflow as tf
import numpy as np
import pandas as pd
import matlab.engine
import matplotlib; matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


import src.TensorFlowModels as TFModels
import src.MotorController as MotorController
from src.ControlSystem import AxisController

"""
IMPORTANT NOTES:

1. Process all the raw data from the simulation into a Matlab file that will smooth and then calculate the step response
characteristics. See:
     https://www.mathworks.com/help/control/ref/stepinfo.html
     https://www.mathworks.com/help/matlab/ref/smoothdata.html
     https://www.mathworks.com/help/matlab/matlab_external/call-user-script-and-function-from-python.html
     https://www.mathworks.com/help/matlab/matlab_external/handle-data-returned-from-matlab-to-python.html
     
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

        self._euler_model = None
        self._euler_cfg = None
        self._gyro_model = None
        self._gyro_cfg = None

        self._euler_graph = tf.Graph()
        self._gyro_graph = tf.Graph()

        self._matlab_engine = None

        self._pitch_ctrl = None
        self._roll_ctrl = None
        self._yaw_ctrl = None

        self._sample_time_mS = 2.0

        self.gyro_symmetric_range_actual = 2000.0       # The actual +/- data range recorded by the gyro
        self.gyro_symmetric_range_mapped = 10.0         # The desired +/- data range input for the NN

        self.motor_range_actual_max = 1860              # ESC input max throttle signal in mS
        self.motor_range_actual_min = 1060              # ESC input min throttle signal in mS

        self.motor_range_mapped_max = 10.0              # NN input max throttle signal (unitless)
        self.motor_range_mapped_min = 0.0               # NN input min throttle signal (unitless)

    def initialize(self, euler_cfg_path, gyro_cfg_path):
        """
        Do things like set up the time series inputs for things like
        step functions, buffer sizes, NN model, Matlab environment etc
        :return:
        """

        # TODO: Add a check for whether or not it is lstm or rnn...i just spent 5 min debugging that...
        # -----------------------------
        # Setup the Euler prediction network
        # -----------------------------
        with self._euler_graph.as_default(), tf.variable_scope('euler'):
            print("Initializing Euler Model...")
            self._euler_cfg = TFModels.ModelConfig()
            self._euler_cfg.load(euler_cfg_path)

            self._euler_model = TFModels.drone_lstm_model(dim_in=self._euler_cfg.input_size,
                                                         dim_out=self._euler_cfg.output_size,
                                                         past_depth=self._euler_cfg.input_depth,
                                                         layer_neurons=self._euler_cfg.neurons_per_layer,
                                                         layer_dropout=self._euler_cfg.layer_dropout,
                                                         learning_rate=self._euler_cfg.learning_rate)

            self._euler_model.load(self._euler_chkpt_path)
            print("Loaded archived Euler model.")

        # -----------------------------
        # Setup the Gyro prediction network
        # -----------------------------
        with self._gyro_graph.as_default(), tf.variable_scope('gyro'):
            print("Initializing Gyro Model...")
            self._gyro_cfg = TFModels.ModelConfig()
            self._gyro_cfg.load(gyro_cfg_path)

            self._gyro_model = TFModels.drone_lstm_model(dim_in=self._gyro_cfg.input_size,
                                                         dim_out=self._gyro_cfg.output_size,
                                                         past_depth=self._gyro_cfg.input_depth,
                                                         layer_neurons=self._gyro_cfg.neurons_per_layer,
                                                         layer_dropout=self._gyro_cfg.layer_dropout,
                                                         learning_rate=self._gyro_cfg.learning_rate)

            self._gyro_model.load(self._gyro_chkpt_path)
            print("Loaded archived Gyro model.")

        # -----------------------------
        # Setup the Matlab Engine
        # -----------------------------
        # print("Starting Matlab Engine")
        # self._matlab_engine = matlab.engine.start_matlab()
        # self._matlab_engine.addpath(r'C:\git\GitHub\ValkyrieRNN\Scripts\Matlab', nargout=0)
        # print("Done")

        # -----------------------------
        # Setup the PID controllers
        # -----------------------------
        self._pitch_ctrl = AxisController(angular_rate_range=100.0,
                                          motor_cmd_range=500.0,
                                          angle_direction=False,
                                          rate_direction=True,
                                          sample_time_ms=self._sample_time_mS)

        self._roll_ctrl = AxisController(angular_rate_range=100.0,
                                         motor_cmd_range=500.0,
                                         angle_direction=True,
                                         rate_direction=True,
                                         sample_time_ms=self._sample_time_mS)

        self._yaw_ctrl = AxisController(angular_rate_range=100.0,
                                        motor_cmd_range=500.0,
                                        angle_direction=True,
                                        rate_direction=True,
                                        sample_time_ms=self._sample_time_mS)

    def TESTFUNC(self):
        print(np.interp(1460,
                        [self.motor_range_actual_min, self.motor_range_actual_max],
                        [self.motor_range_mapped_min, self.motor_range_mapped_max]))

        print("Test of vector interpolation")
        print(np.interp([[-2000, -1000, 0.0, 100, 1000, 2000], [-2000, -1000, 0.0, 100, 1000, 2000]],
                        [-self.gyro_symmetric_range_actual, self.gyro_symmetric_range_actual],
                        [-self.gyro_symmetric_range_mapped, self.gyro_symmetric_range_mapped]))

    def set_pitch_ctrl_pid(self, kp_angle, ki_angle, kd_angle, kp_rate, ki_rate, kd_rate):
        self._pitch_ctrl.update_angle_pid(kp_angle, ki_angle, kd_angle)
        self._pitch_ctrl.update_rate_pid(kp_rate, ki_rate, kd_rate)

    def set_roll_ctrl_pid(self, kp_angle, ki_angle, kd_angle, kp_rate, ki_rate, kd_rate):
        self._roll_ctrl.update_angle_pid(kp_angle, ki_angle, kd_angle)
        self._roll_ctrl.update_rate_pid(kp_rate, ki_rate, kd_rate)

    def set_yaw_ctrl_pid(self, kp_angle, ki_angle, kd_angle, kp_rate, ki_rate, kd_rate):
        self._yaw_ctrl.update_angle_pid(kp_angle, ki_angle, kd_angle)
        self._yaw_ctrl.update_rate_pid(kp_rate, ki_rate, kd_rate)

    @property
    def pitch_pid(self):
        return dict({'angles': [self._pitch_ctrl.angleController.kp,
                                self._pitch_ctrl.angleController.ki,
                                self._pitch_ctrl.angleController.kd],
                     'rates':  [self._pitch_ctrl.rateController.kp,
                                self._pitch_ctrl.rateController.ki,
                                self._pitch_ctrl.rateController.kd]})

    @property
    def roll_pid(self):
        return dict({'angles': [self._roll_ctrl.angleController.kp,
                                self._roll_ctrl.angleController.ki,
                                self._roll_ctrl.angleController.kd],
                     'rates': [self._roll_ctrl.rateController.kp,
                               self._roll_ctrl.rateController.ki,
                               self._roll_ctrl.rateController.kd]})

    @property
    def yaw_pid(self):
        return dict({'angles': [self._yaw_ctrl.angleController.kp,
                                self._yaw_ctrl.angleController.ki,
                                self._yaw_ctrl.angleController.kd],
                     'rates': [self._yaw_ctrl.rateController.kp,
                               self._yaw_ctrl.rateController.ki,
                               self._yaw_ctrl.rateController.kd]})

    def simulate_pitch_step(self, step_input_delta, step_enable_t0, num_sim_steps):
        """

        :param step_input_delta:
        :param step_enable_t0:
        :param num_sim_steps:
        :return: (1 x sim_length) ndarray of pitch response
        """
        assert(np.isscalar(step_input_delta))
        assert(np.isscalar(step_enable_t0))
        assert(np.isscalar(num_sim_steps))

        # Input history indices
        asp_idx = 0     # Angle setpoint Pitch
        asr_idx = 1     # Angle setpoint Roll
        asy_idx = 2     # Angle setpoint Yaw

        # Output history indices
        pit_idx = 0     # Pitch Angle
        rol_idx = 1     # Roll Angle
        gx_idx = 0      # MEMS Gyro X axis rotation rate
        gy_idx = 1      # MEMS Gyro Y axis rotation rate

        # Network dimensions
        dim_in = self._euler_cfg.input_size         # == self._gyro_cfg.input_size
        dim_depth = self._euler_cfg.input_depth     # == self._gyro_cfg.input_depth
        dim_out = self._euler_cfg.output_size       # == self._gyro_cfg.output_size

        # Buffers for working with the NN
        input_history = np.zeros([dim_in, dim_depth])       # Latest value is in FIRST column [:, 0}
        euler_output_history = np.zeros([dim_out, 1])       # Latest value is LAST [:, -1]
        gyro_output_history = np.zeros([dim_out, 1])        # Latest value is LAST [:, -1]

        angle_setpoint_history = np.zeros([3, num_sim_steps])

        base_throttle = 1160.0

        # -----------------------------------
        # Run the full simulation of the input signals
        # -----------------------------------
        print("Starting simulation of pitch step input...")
        start_time = time.perf_counter()

        for sim_step in range(0, num_sim_steps):

            if sim_step >= step_enable_t0:
                angle_setpoint_history[asp_idx, sim_step] = step_input_delta

            # -----------------------------------
            # Generate a new motor signal
            # -----------------------------------
            pitch_cmd = self._step_pitch_controller(
                angle_setpoint_history[asp_idx, sim_step],  # Last pitch angle cmd input
                euler_output_history[pit_idx, -1],          # Last computed pitch angle
                gyro_output_history[gy_idx, -1])            # Last computed pitch rate

            roll_cmd = self._step_roll_controller(
                angle_setpoint_history[asr_idx, sim_step],  # Last roll angle cmd input
                euler_output_history[rol_idx, -1],          # Last computed roll angle
                -gyro_output_history[gx_idx, -1])           # Last computed roll rate

            yaw_cmd = 0

            motor_signal = MotorController.generate_motor_signals(base_throttle, pitch_cmd, roll_cmd, yaw_cmd)

            # -----------------------------------
            # Generate a new NN input
            # -----------------------------------
            new_input_column = np.r_[motor_signal[0],
                                     motor_signal[1],
                                     motor_signal[2],
                                     motor_signal[3]].reshape(dim_in, 1)
            assert(np.shape(new_input_column) == (dim_in, 1)), "Your model input is the wrong shape!!"

            # Roll the time history column-wise so that the oldest value pops up @ index zero
            input_history = np.roll(input_history, 1, axis=1)

            # Replace the now old column with the latest
            input_history[:, 0] = new_input_column[:, 0]

            # Expands matrix dimensions for NN compatibility
            nn_motor_input = np.expand_dims(input_history, axis=0)
            assert(np.shape(nn_motor_input) == (1, dim_in, dim_depth)), "You realize your depth size is last right?"

            # -----------------------------------
            # Map the input data into the range expected by the NN
            # -----------------------------------
            nn_motor_input = self._real_motor_data_to_mapped(nn_motor_input)

            # -----------------------------------
            # Predict new Euler/Gyro outputs
            # -----------------------------------
            # Generate predictions in the training space
            euler_data = self._euler_model.predict(nn_motor_input)
            gyro_data = self._gyro_model.predict(nn_motor_input)

            # Convert training space data to real world data
            gyro_data = self._mapped_gyro_data_to_real(gyro_data)

            # -----------------------------------
            # Update the output buffers with the latest predictions
            # -----------------------------------
            new_euler_output_column = np.array([euler_data[0][pit_idx], euler_data[0][rol_idx]]).reshape(2, 1)
            euler_output_history = np.append(euler_output_history, new_euler_output_column, axis=1)

            new_gyro_output_column = np.array([gyro_data[0][gx_idx], gyro_data[0][gy_idx]]).reshape(2, 1)
            gyro_output_history = np.append(gyro_output_history, new_gyro_output_column, axis=1)

        end_time = time.perf_counter()
        elapsed_time = end_time-start_time
        print("Total Time: ", elapsed_time)

        return euler_output_history, gyro_output_history, angle_setpoint_history, input_history

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

    def _smooth_raw_output(self):
        """
        Implements a smoothing filter so that we don't have so much noise to
        process
        :return:
        """
        raise NotImplementedError

    def _parse_model_config(self):
        raise NotImplementedError

    def _step_pitch_controller(self, angle_setpoint, angle_feedback, rate_feedback):
        self._pitch_ctrl.angle_setpoint = angle_setpoint
        self._pitch_ctrl.angle_feedback = angle_feedback
        self._pitch_ctrl.rate_feedback = rate_feedback

        self._pitch_ctrl.compute()

        return self._pitch_ctrl.controller_output

    def _step_roll_controller(self, angle_setpoint, angle_feedback, rate_feedback):
        self._roll_ctrl.angle_setpoint = angle_setpoint
        self._roll_ctrl.angle_feedback = angle_feedback
        self._roll_ctrl.rate_feedback = rate_feedback

        self._roll_ctrl.compute()

        return self._roll_ctrl.controller_output

    def _step_yaw_controller(self, angle_setpoint, angle_feedback, rate_feedback):
        self._yaw_ctrl.angle_setpoint = angle_setpoint
        self._yaw_ctrl.angle_feedback = angle_feedback
        self._yaw_ctrl.rate_feedback = rate_feedback

        self._yaw_ctrl.compute()

        return self._yaw_ctrl.controller_output

    def _mapped_gyro_data_to_real(self, input_signal):
        return np.interp(input_signal,
                         [-self.gyro_symmetric_range_mapped, self.gyro_symmetric_range_mapped],
                         [-self.gyro_symmetric_range_actual, self.gyro_symmetric_range_actual])

    def _real_gyro_data_to_mapped(self, input_signal):
        return np.interp(input_signal,
                         [-self.gyro_symmetric_range_actual, self.gyro_symmetric_range_actual],
                         [-self.gyro_symmetric_range_mapped, self.gyro_symmetric_range_mapped])

    def _mapped_motor_data_to_real(self, input_signal):
        return np.interp(input_signal,
                         [self.motor_range_mapped_min, self.motor_range_mapped_max],
                         [self.motor_range_actual_min, self.motor_range_actual_max])

    def _real_motor_data_to_mapped(self, input_signal):
        return np.interp(input_signal,
                         [self.motor_range_actual_min, self.motor_range_actual_max],
                         [self.motor_range_mapped_min, self.motor_range_mapped_max])


if __name__ == "__main__":
    print('Hello')
