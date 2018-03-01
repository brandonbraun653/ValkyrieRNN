from __future__ import division, print_function, absolute_import

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

        self._sample_time_mS = 2.0

    def initialize(self, euler_cfg_path, gyro_cfg_path):
        """
        Do things like set up the time series inputs for things like
        step functions, buffer sizes, NN model, Matlab environment etc
        :return:
        """

        # Setup the Euler prediction network
        with self._euler_graph.as_default(), tf.variable_scope('euler'):
            print("Initializing Euler Model...")
            euler_cfg = TFModels.ModelConfig()
            euler_cfg.load(euler_cfg_path)

            # TODO: Add a check for whether or not it is lstm or rnn...i just spent 5 min debugging that...
            self._euler_model = TFModels.drone_lstm_model(dim_in=euler_cfg.input_size,
                                                         dim_out=euler_cfg.output_size,
                                                         past_depth=euler_cfg.input_depth,
                                                         layer_neurons=euler_cfg.neurons_per_layer,
                                                         layer_dropout=euler_cfg.layer_dropout,
                                                         learning_rate=euler_cfg.learning_rate)

            self._euler_model.load(self._euler_chkpt_path)
            print("Loaded archived Euler model.")

        # Setup the Gyro prediction network
        with self._gyro_graph.as_default(), tf.variable_scope('gyro'):
            print("Initializing Gyro Model...")
            gyro_cfg = TFModels.ModelConfig()
            gyro_cfg.load(gyro_cfg_path)

            self._gyro_model = TFModels.drone_lstm_model(dim_in=gyro_cfg.input_size,
                                                         dim_out=gyro_cfg.output_size,
                                                         past_depth=gyro_cfg.input_depth,
                                                         layer_neurons=gyro_cfg.neurons_per_layer,
                                                         layer_dropout=gyro_cfg.layer_dropout,
                                                         learning_rate=gyro_cfg.learning_rate)

            self._gyro_model.load(self._gyro_chkpt_path)
            print("Loaded archived Gyro model.")

        # Setup the Matlab environment
        # print("Starting Matlab Engine")
        # self._matlab_engine = matlab.engine.start_matlab()
        # self._matlab_engine.addpath(r'C:\git\GitHub\ValkyrieRNN\Scripts\Matlab', nargout=0)
        # print("Done")

        # Setup the PID controllers
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
        timeSeries = pd.read_csv('DroneData/csv/timeSeriesInferenceDataSmoothed.csv')
        # timeSeries = pd.read_csv('DroneData/csv/timeSeriesDataInterpolated.csv')

        angles = np.array([timeSeries['pitch'],
                           timeSeries['roll'],
                           timeSeries['yaw']]).transpose()

        rates = np.array([timeSeries['gx'],
                          timeSeries['gy'],
                          timeSeries['gz']]).transpose()

        angle_setpoints = np.array([timeSeries['asp'],
                                    timeSeries['asr'],
                                    timeSeries['asy']]).transpose()

        rate_setpoints = np.array([timeSeries['rsp'],
                                   timeSeries['rsr'],
                                   timeSeries['rsy']]).transpose()

        motor_outputs = np.array([timeSeries['m1CMD'],
                                  timeSeries['m2CMD'],
                                  timeSeries['m3CMD'],
                                  timeSeries['m4CMD']]).transpose()

        # Column accessor indices
        pitch = 0
        roll = 1
        yaw = 2

        samples = 100000

        actual = {'angle_setpoint': {'pitch': [], 'roll': [], 'yaw': []},
                  'angle_feedback': {'pitch': [], 'roll': [], 'yaw': []},
                  'rate_setpoint': {'pitch': [], 'roll': [], 'yaw': []},
                  'rate_feedback': {'pitch': [], 'roll': [], 'yaw': []},
                  'motor_output': {'m1': [], 'm2': [], 'm3': [], 'm4': []}}

        predicted = {'angle_setpoint': {'pitch': [], 'roll': [], 'yaw': []},
                     'angle_feedback': {'pitch': [], 'roll': [], 'yaw': []},
                     'rate_setpoint': {'pitch': [], 'roll': [], 'yaw': []},
                     'rate_feedback': {'pitch': [], 'roll': [], 'yaw': []},
                     'motor_output': {'m1': [], 'm2': [], 'm3': [], 'm4': []}}

        base_throttle = 1160.0
        for val in range(0, samples):
            # Update PID controller inputs
            self._pitch_ctrl.angle_setpoint = angle_setpoints[val, pitch]
            self._pitch_ctrl.angle_feedback = angles[val, pitch]
            self._pitch_ctrl.rate_feedback = rates[val, 1]                  # Pitch about +Y axis

            self._roll_ctrl.angle_setpoint = angle_setpoints[val, roll]
            self._roll_ctrl.angle_feedback = angles[val, roll]
            self._roll_ctrl.rate_feedback = -1.0*rates[val, 0]              # Roll about -X axis

            self._yaw_ctrl.angle_setpoint = angle_setpoints[val, yaw]
            self._yaw_ctrl.angle_feedback = angles[val, yaw]
            self._yaw_ctrl.rate_feedback = rates[val, 2]                    # Yaw about Z axis

            # Update the controller output
            self._pitch_ctrl.compute()
            self._roll_ctrl.compute()
            self._yaw_ctrl.compute()

            # Update the motor outputs
            pitch_cmd = self._pitch_ctrl.controller_output
            roll_cmd = self._roll_ctrl.controller_output
            yaw_cmd = self._yaw_ctrl.controller_output

            motor_signal = MotorController.generate_motor_signals(base_throttle, pitch_cmd, roll_cmd, yaw_cmd)

            # Log for plotting later
            predicted['angle_setpoint']['pitch'].append(self._pitch_ctrl.angle_setpoint)
            predicted['angle_setpoint']['roll'].append(self._roll_ctrl.angle_setpoint)
            predicted['angle_setpoint']['yaw'].append(self._yaw_ctrl.angle_setpoint)

            actual['angle_setpoint']['pitch'].append(angle_setpoints[val, pitch])
            actual['angle_setpoint']['roll'].append(angle_setpoints[val, roll])
            actual['angle_setpoint']['yaw'].append(angle_setpoints[val, yaw])

            predicted['rate_setpoint']['pitch'].append(self._pitch_ctrl.angular_rate_desired)
            actual['rate_setpoint']['pitch'].append(rate_setpoints[val, pitch])

            predicted['motor_output']['m1'].append(motor_signal[0])
            predicted['motor_output']['m2'].append(motor_signal[1])
            predicted['motor_output']['m3'].append(motor_signal[2])
            predicted['motor_output']['m4'].append(motor_signal[3])

            actual['motor_output']['m1'].append(motor_outputs[val, 0])
            actual['motor_output']['m2'].append(motor_outputs[val, 1])
            actual['motor_output']['m3'].append(motor_outputs[val, 2])
            actual['motor_output']['m4'].append(motor_outputs[val, 3])


        # plt.figure(figsize=(16, 4))
        # plt.suptitle('Pitch Angle')
        # plt.plot(predicted['angle_setpoint']['pitch'], label='Pitch Py')
        # plt.plot(predicted['angle_setpoint']['roll'], label='Roll Py')
        # plt.plot(predicted['angle_setpoint']['yaw'], label='Yaw Py')
        # plt.plot(actual['angle_setpoint']['pitch'], label='Pitch Drone')
        # plt.plot(actual['angle_setpoint']['roll'], label='Roll Drone')
        # plt.plot(actual['angle_setpoint']['yaw'], label='Yaw Drone')
        # plt.legend()

        # plt.figure(figsize=(16, 4))
        # plt.suptitle('Pitch Actual Vs Predicted')
        # plt.plot(predicted['motor_output']['m1'], 'c-', label='M1 Predicted')
        # plt.plot(actual['motor_output']['m1'], 'b-', label='M1 Actual')
        # plt.legend()

        plt.figure(figsize=(16, 4))
        plt.suptitle('Pitch Rate Actual Vs Predicted')
        plt.plot(predicted['rate_setpoint']['pitch'], 'c-', label='Pitch Rate Predicted')
        plt.plot(actual['rate_setpoint']['pitch'], 'b-', label='Pitch Rate Actual')
        plt.legend()
        plt.show()

    def set_pitch_ctrl_pid(self, kp_angle, ki_angle, kd_angle, kp_rate, ki_rate, kd_rate):
        self._pitch_ctrl.update_angle_pid(kp_angle, ki_angle, kd_angle)
        self._pitch_ctrl.update_rate_pid(kp_rate, ki_rate, kd_rate)

    def set_roll_ctrl_pid(self, kp_angle, ki_angle, kd_angle, kp_rate, ki_rate, kd_rate):
        self._roll_ctrl.update_angle_pid(kp_angle, ki_angle, kd_angle)
        self._roll_ctrl.update_rate_pid(kp_rate, ki_rate, kd_rate)

    def set_yaw_ctrl_pid(self, kp_angle, ki_angle, kd_angle, kp_rate, ki_rate, kd_rate):
        self._yaw_ctrl.update_angle_pid(kp_angle, ki_angle, kd_angle)
        self._yaw_ctrl.update_rate_pid(kp_rate, ki_rate, kd_rate)

    def simulate_pitch_step(self, step_size, sim_length):
        """

        :param step_size:
        :param sim_length:
        :return: (1 x sim_length) ndarray of pitch response
        """

        step_time_threshold = 5

        # Input history indices
        asp_idx = 4     # Angle setpoint Pitch
        asr_idx = 5     # Angle setpoint Roll

        # Output history indices
        pitch_angle_idx = 0
        roll_angle_idx = 1
        gx_idx = 2
        gy_idx = 3


        # TODO: Take these values from the NN configuration files
        input_history = np.zeros([10, 6])       # Latest value is FIRST
        output_history = np.zeros([4, 1])       # Latest value is LAST

        base_throttle = 1160.0

        for sim_step in range(0, sim_length):

            # Enable the step input
            if sim_step >= step_time_threshold:
                input_history[asp_idx, 0] = step_size

            # -----------------------------------
            # Generate a new motor signal
            # -----------------------------------
            pitch_cmd = self._step_pitch_controller(input_history[asp_idx, 0],              # Last pitch angle cmd input
                                                    output_history[pitch_angle_idx, -1],    # Last computed pitch angle
                                                    output_history[gy_idx, -1])             # Last computed pitch rate

            roll_cmd = self._step_roll_controller(input_history[asr_idx, 0],                # Last roll angle cmd input
                                                  output_history[roll_angle_idx, -1],       # Last computed roll angle
                                                 -output_history[gx_idx, -1])               # Last computed roll rate

            yaw_cmd = 0

            motor_signal = MotorController.generate_motor_signals(base_throttle, pitch_cmd, roll_cmd, yaw_cmd)

            # -----------------------------------
            # Generate a new NN input
            # -----------------------------------
            new_input_column = np.r_[motor_signal[0],
                                     motor_signal[1],
                                     motor_signal[2],
                                     motor_signal[3],
                                     input_history[asp_idx, 0],
                                     input_history[asr_idx, 0]].reshape(6, 1)
            assert(np.shape(new_input_column) == (6, 1))

            # Roll the time history array so that the oldest value pops up @ index zero
            input_history = np.roll(input_history, 1, axis=1)

            # Replace the now old value with the latest value
            input_history[:, 0] = new_input_column[:, 0]

            # -----------------------------------
            # Pass the input to the Euler and Gyro networks
            # -----------------------------------
            # Formats the data into a [10, 6] matrix
            X = input_history.transpose()

            # Expands matrix dimensions to be [1, 10, 6] for NN compatibility
            X = np.expand_dims(input_history, axis=0)

            revert back to 6 x N input size...

            euler_data = self._euler_model.predict(X)
            gyro_data = self._gyro_model.predict(X)

            # model predict....x2
            _gx = gyro_data[0][0]
            _gy = gyro_data[0][1]
            _pitch = euler_data[0][0]
            _roll = euler_data[0][1]
            new_output_column = np.array([_pitch, _roll, _gx, _gy]).reshape(4, 1)

            output_history = np.append(output_history, new_output_column, axis=1)

        return output_history





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



if __name__ == "__main__":
    print('Hello')
