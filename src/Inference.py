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
from src.Miscellaneous import bcolors

from Scripts.Matlab.MatlabIOHelper import matlab_matrix_to_numpy, numpy_matrix_to_matlab
from src.TensorFlowModels import ModelConfig

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
    def __init__(self, euler_cfg_path='', gyro_cfg_path=''):
        self._euler_cfg_path = euler_cfg_path
        self._euler_cfg = ModelConfig()
        self._euler_model = None
        self._euler_graph = tf.Graph()
        self._euler_input_shape = None

        self._gyro_cfg_path = gyro_cfg_path
        self._gyro_cfg = ModelConfig()
        self._gyro_model = None
        self._gyro_graph = tf.Graph()
        self._gyro_input_shape = None

        self._matlab_engine = None

        self._pitch_ctrl = None
        self._roll_ctrl = None
        self._yaw_ctrl = None

        self._pid_sample_time_mS = 8.0  # Valkyrie FCS pid controller update rate
        self._sim_dt = 0.002

        # TODO: Update these externally somehow
        self.gyro_symmetric_range_actual = 250.0       # The actual +/- data range recorded by the gyro
        self.gyro_symmetric_range_mapped = 1.0         # The desired +/- data range input for the NN
        self.motor_range_actual_max = 1860             # ESC input max throttle signal in mS
        self.motor_range_actual_min = 1060             # ESC input min throttle signal in mS
        self.motor_range_mapped_max = 1.0              # NN input max throttle signal (unitless)
        self.motor_range_mapped_min = 0.0              # NN input min throttle signal (unitless)

    def initialize(self, ahrs_sample_freq, pid_update_freq):
        """
        Do things like set up the time series inputs for things like
        step functions, buffer sizes, NN model, Matlab environment etc
        :return:
        """
        self._pid_sample_time_mS = int((1.0 / pid_update_freq) * 1000.0)
        self._sim_dt = 1.0 / ahrs_sample_freq

        # -----------------------------
        # Setup the Euler prediction network
        # -----------------------------
        self._euler_cfg.load(self._euler_cfg_path)
        with self._euler_graph.as_default(), tf.variable_scope('euler'):
            print(bcolors.OKGREEN + 'Initializing Euler Model' + bcolors.ENDC)

            # Make sure the input shape matches the cfg file
            if self._euler_cfg.data_inversion:
                self._euler_input_shape = [None, self._euler_cfg.input_size, self._euler_cfg.input_depth]
            else:
                self._euler_input_shape = [None, self._euler_cfg.input_depth, self._euler_cfg.input_size]

            # Select the proper model type
            if 'lstm_model_deep' in self._euler_cfg.model_type:
                self._euler_model = TFModels.drone_lstm_model_deep(shape=self._euler_input_shape,
                                                                   dim_in=self._euler_cfg.input_size,
                                                                   dim_out=self._euler_cfg.output_size,
                                                                   past_depth=self._euler_cfg.input_depth,
                                                                   layer_neurons=self._euler_cfg.neurons_per_layer,
                                                                   layer_dropout=self._euler_cfg.layer_dropout,
                                                                   learning_rate=self._euler_cfg.learning_rate)
            elif 'rnn' in self._euler_cfg.model_type:
                self._euler_model = TFModels.drone_rnn_model(shape=self._euler_input_shape,
                                                             dim_in=self._euler_cfg.input_size,
                                                             dim_out=self._euler_cfg.output_size,
                                                             past_depth=self._euler_cfg.input_depth,
                                                             layer_neurons=self._euler_cfg.neurons_per_layer,
                                                             layer_dropout=self._euler_cfg.layer_dropout,
                                                             learning_rate=self._euler_cfg.learning_rate)
            elif 'deeply_connected' in self._euler_cfg.model_type:
                self._euler_model = TFModels.drone_lstm_deeply_connected(shape=self._euler_input_shape,
                                                                         dim_in=self._euler_cfg.input_size,
                                                                         dim_out=self._euler_cfg.output_size,
                                                                         past_depth=self._euler_cfg.input_depth,
                                                                         layer_neurons=self._euler_cfg.neurons_per_layer,
                                                                         layer_dropout=self._euler_cfg.layer_dropout,
                                                                         learning_rate=self._euler_cfg.learning_rate)

            # TODO: Somehow I need to log the "extension" code for these model checkpoints
            self._euler_model.load(self._euler_cfg.best_chkpt_path + '7871')
            print(bcolors.OKGREEN + "Loaded archived Euler model" + bcolors.ENDC)

            # Reset the input shape for use in numpy matrix initializations later: None -> -1
            # Hard coding is ok due to TF framework requiring the 'None' keyword always be the first
            # value in the input shape list. See: http://tflearn.org/layers/core/
            self._euler_input_shape[0] = -1

        # -----------------------------
        # Setup the Gyro prediction network
        # -----------------------------
        self._gyro_cfg.load(self._gyro_cfg_path)
        with self._gyro_graph.as_default(), tf.variable_scope('gyro'):
            print(bcolors.OKGREEN + "Initializing Gyro Model..." + bcolors.ENDC)

            # Make sure the input shape matches the cfg file
            if self._gyro_cfg.data_inversion:
                self._gyro_input_shape = [None, self._gyro_cfg.input_size, self._gyro_cfg.input_depth]
            else:
                self._gyro_input_shape = [None, self._gyro_cfg.input_depth, self._gyro_cfg.input_size]

            # Select the proper model type
            if 'lstm_model_deep' in self._gyro_cfg.model_type:
                self._gyro_model = TFModels.drone_lstm_model_deep(shape=self._gyro_input_shape,
                                                                  dim_in=self._gyro_cfg.input_size,
                                                                  dim_out=self._gyro_cfg.output_size,
                                                                  past_depth=self._gyro_cfg.input_depth,
                                                                  layer_neurons=self._gyro_cfg.neurons_per_layer,
                                                                  layer_dropout=self._gyro_cfg.layer_dropout,
                                                                  learning_rate=self._gyro_cfg.learning_rate)
            elif 'rnn' in self._gyro_cfg.model_type:
                self._gyro_model = TFModels.drone_rnn_model(shape=self._gyro_input_shape,
                                                            dim_in=self._gyro_cfg.input_size,
                                                            dim_out=self._gyro_cfg.output_size,
                                                            past_depth=self._gyro_cfg.input_depth,
                                                            layer_neurons=self._gyro_cfg.neurons_per_layer,
                                                            layer_dropout=self._gyro_cfg.layer_dropout,
                                                            learning_rate=self._gyro_cfg.learning_rate)
            elif 'deeply_connected' in self._gyro_cfg.model_type:
                self._gyro_model = TFModels.drone_lstm_deeply_connected(shape=self._gyro_input_shape,
                                                                        dim_in=self._gyro_cfg.input_size,
                                                                        dim_out=self._gyro_cfg.output_size,
                                                                        past_depth=self._gyro_cfg.input_depth,
                                                                        layer_neurons=self._gyro_cfg.neurons_per_layer,
                                                                        layer_dropout=self._gyro_cfg.layer_dropout,
                                                                        learning_rate=self._gyro_cfg.learning_rate)

            # TODO: Somehow I need to log the "extension" code for these model checkpoints
            self._gyro_model.load(self._gyro_cfg.best_chkpt_path + '5622')
            print(bcolors.OKGREEN + "Loaded archived Gyro model" + bcolors.ENDC)

            # Reset the input shape for use in numpy matrix initializations later: None -> -1
            # Hard coding is ok due to TF framework requiring the 'None' keyword always be the first
            # value in the input shape list. See: http://tflearn.org/layers/core/
            self._gyro_input_shape[0] = -1

        # -----------------------------
        # Setup the Matlab Engine
        # -----------------------------
        print(bcolors.OKGREEN + "Starting Matlab Engine" + bcolors.ENDC)
        self._matlab_engine = matlab.engine.start_matlab()
        self._matlab_engine.addpath(r'C:\git\GitHub\ValkyrieRNN\Scripts\Matlab', nargout=0)

        # -----------------------------
        # Setup the PID controllers
        # -----------------------------
        # TODO: Eventually take these configuration parameters from a JSON file that links up with the
        # Valkyrie FCS C++ source code
        print(bcolors.OKGREEN + "Initializing PID Controllers" + bcolors.ENDC)
        self._pitch_ctrl = AxisController(angular_rate_range=100.0,
                                          motor_cmd_range=500.0,
                                          angle_direction=True,
                                          rate_direction=True,
                                          sample_time_ms=self._pid_sample_time_mS)

        self._roll_ctrl = AxisController(angular_rate_range=100.0,
                                         motor_cmd_range=500.0,
                                         angle_direction=True,
                                         rate_direction=False,
                                         sample_time_ms=self._pid_sample_time_mS)

        self._yaw_ctrl = AxisController(angular_rate_range=100.0,
                                        motor_cmd_range=500.0,
                                        angle_direction=True,
                                        rate_direction=True,
                                        sample_time_ms=self._pid_sample_time_mS)

    def simulate_pitch_step(self, start_time, end_time, step_input_delta, step_ON_pct=0.5):
        num_sim_steps = int(np.ceil((end_time-start_time)/self._sim_dt))
        time_history = np.linspace(start_time, end_time, num_sim_steps)

        # -----------------------------------
        # Setup a few configuration variables
        # -----------------------------------
        base_throttle = 1160.0

        # Input history indices
        asp_idx = 0     # Angle setpoint Pitch
        asr_idx = 1     # Angle setpoint Roll
        asy_idx = 2     # Angle setpoint Yaw

        # Output history indices
        pit_idx = 0     # Pitch Angle
        rol_idx = 1     # Roll Angle
        gx_idx = 0      # MEMS Gyro X axis rotation rate
        gy_idx = 1      # MEMS Gyro Y axis rotation rate

        # Latest value is in LAST column [:, -1]
        euler_input_history = np.zeros([self._euler_input_shape[1], self._euler_input_shape[2]])
        gyro_input_history = np.zeros([self._gyro_input_shape[1], self._gyro_input_shape[2]])

        # Latest value is LAST [:, -1]. Data appended as simulation executes
        euler_output_history = np.zeros([self._euler_cfg.output_size, 1])
        gyro_output_history = np.zeros([self._gyro_cfg.output_size, 1])

        # Inputs to the pid controllers
        angle_setpoint_history = np.zeros([3, num_sim_steps])

        # Logging
        euler_cmd_history = np.zeros([4, num_sim_steps])
        gyro_cmd_history = np.zeros([4, num_sim_steps])
        pitch_cmd_history = []
        roll_cmd_history = []
        yaw_cmd_history = []

        # -----------------------------------
        # Run the full simulation of the input signals
        # -----------------------------------
        print(bcolors.OKGREEN + "Starting simulation of pitch step input" + bcolors.ENDC)
        start_time = time.perf_counter()

        pitch_cmd = 0.0
        roll_cmd = 0.0
        yaw_cmd = 0.0

        sim_step = 0
        current_time = 0.0
        last_time = 0.0

        while sim_step < num_sim_steps-1:

            if sim_step >= int(step_ON_pct*num_sim_steps):
                angle_setpoint_history[asp_idx, sim_step] = step_input_delta

            # -----------------------------------
            # Generate a new motor signal
            # -----------------------------------
            # Only update the pid controller at the correct frequency
            if (current_time - last_time) > (self._pid_sample_time_mS/1000.0):
                last_time = current_time

                pitch_cmd = self._step_pitch_controller(
                    angle_setpoint_history[asp_idx, sim_step],   # Last pitch angle cmd input
                    euler_output_history[pit_idx, -1],           # Last computed pitch angle
                    gyro_output_history[gy_idx, -1])             # Last computed pitch rate

                roll_cmd = self._step_roll_controller(
                    angle_setpoint_history[asr_idx, sim_step],   # Last roll angle cmd input
                    euler_output_history[rol_idx, -1],           # Last computed roll angle
                    -gyro_output_history[gx_idx, -1])            # Last computed roll rate

                yaw_cmd = 0

            motor_signal = MotorController.generate_motor_signals(base_throttle, pitch_cmd, roll_cmd, yaw_cmd)

            pitch_cmd_history.append(pitch_cmd)
            roll_cmd_history.append(roll_cmd)
            yaw_cmd_history.append(yaw_cmd)

            # -----------------------------------
            # Generate a new NN input
            # -----------------------------------
            new_input_column = np.r_[motor_signal[0],
                                     motor_signal[1],
                                     motor_signal[2],
                                     motor_signal[3]].reshape(self._euler_cfg.input_size, 1)

            # Add some noise to help excite internal mode dynamics
            noise = np.random.uniform(0.0, 1.0, [self._euler_cfg.input_size, 1])
            new_input_column += noise

            # Roll the time history column-wise so that the oldest value [:,0] pops up at [:,-1]
            euler_input_history = np.roll(euler_input_history, -1, axis=1)
            gyro_input_history = np.roll(gyro_input_history, -1, axis=1)

            # Update the most recent input
            euler_input_history[:, -1] = new_input_column[:, 0]
            gyro_input_history[:, -1] = new_input_column[:, 0]

            # -----------------------------------
            # Map the input data into the range expected by the NN
            # -----------------------------------
            euler_input = self._real_motor_data_to_mapped(euler_input_history)
            gyro_input = self._real_gyro_data_to_mapped(gyro_input_history)

            euler_cmd_history[:, sim_step] = euler_input[:, -1]
            gyro_cmd_history[:, sim_step] = gyro_input[:, -1]

            # -----------------------------------
            # Predict new Euler/Gyro outputs
            # This correlates to the ahrs update rate
            # -----------------------------------
            # TODO: Going to need to account for inverted input shapes here sooner or later...
            # Expand so that the 0th axis tells the NN what our batch size is: aka 1.
            e_nn_in = np.expand_dims(euler_input, axis=0)
            g_nn_in = np.expand_dims(gyro_input, axis=0)

            # Generate predictions in the training space
            euler_prediction = self._euler_model.predict(e_nn_in)
            gyro_prediction = self._gyro_model.predict(g_nn_in)

            # Convert training space data to real world data
            # gyro_prediction = self._mapped_gyro_data_to_real(gyro_prediction)
            # TODO: SHOULD ALREADY BE IN REAL GYRO SPACE

            # -----------------------------------
            # Update the output buffers with the latest predictions
            # -----------------------------------
            new_euler_output_column = np.array([euler_prediction[0][pit_idx], euler_prediction[0][rol_idx]]).reshape(2, 1)
            euler_output_history = np.append(euler_output_history, new_euler_output_column, axis=1)

            new_gyro_output_column = np.array([gyro_prediction[0][gx_idx], gyro_prediction[0][gy_idx]]).reshape(2, 1)
            gyro_output_history = np.append(gyro_output_history, new_gyro_output_column, axis=1)

            # -----------------------------------
            # Update simulation variables
            # -----------------------------------
            sim_step += 1
            current_time += self._sim_dt

        end_time = time.perf_counter()
        elapsed_time = end_time-start_time
        print("Total Time: ", elapsed_time)

        results = {'euler_output': euler_output_history,
                   'gyro_output': gyro_output_history,
                   'time': time_history,
                   'angle_setpoints': angle_setpoint_history,
                   'euler_input': euler_cmd_history,
                   'gyro_input': gyro_cmd_history,
                   'pitch_ctrl_output': pitch_cmd_history,
                   'roll_ctrl_output': roll_cmd_history,
                   'yaw_ctrl_output': yaw_cmd_history}

        return results

    def simulate_roll_step(self, step_size, sim_length):
        raise NotImplementedError

    def simulate_coupled_pitch_roll_step(self, step_size_pitch, step_size_roll, sim_length):
        raise NotImplementedError

    def analyze_step_performance_siso(self, input_data, expected_final_value, start_time, end_time):
        assert(input_data.ndim == 1)
        assert(np.isscalar(expected_final_value))
        assert(np.isscalar(start_time))
        assert(np.isscalar(end_time))

        time_len = np.shape(input_data)
        input_mdt = numpy_matrix_to_matlab(input_data)
        time_mdt = numpy_matrix_to_matlab(np.linspace(start_time, end_time, time_len[0]))
        return self._matlab_engine.CalculateStepPerformance(input_mdt, time_mdt, expected_final_value)

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
