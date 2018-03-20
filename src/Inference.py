from __future__ import division, print_function, absolute_import
import os
import glob
import time
import json
import tflearn

import tensorflow as tf
import numpy as np
import pandas as pd
import matlab.engine
import matplotlib; matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import Scripts.RawData2CSV as Converter
import src.TensorFlowModels as TFModels
import src.MotorController as MotorController

from pprint import pprint
from src.DataHandling import DataHandler
from src.ControlSystem import AxisController
from src.Miscellaneous import bcolors

from Scripts.Matlab.MatlabIOHelper import matlab_matrix_to_numpy, numpy_matrix_to_matlab
from src.TensorFlowModels import ModelConfig


def is_locked(filepath):
    """Checks if a file is locked by opening it in append mode.
    If no exception thrown, then the file is not locked.
    """
    locked = None
    file_object = None
    if os.path.exists(filepath):
        try:
            print("Trying to open", filepath)
            buffer_size = 8
            # Opening file in append mode and read the first 8 characters.
            file_object = open(filepath, 'a', buffer_size)
            if file_object:
                print(filepath, "is not locked.")
                locked = False
        except IOError:
            print("File is locked (unable to open in append mode).")
            locked = True
        finally:
            if file_object:
                file_object.close()
    else:
        print(filepath, " not found.")
    return locked


class ModelInferencer(DataHandler):
    def __init__(self, config_path=None, model_checkpoint=None, data_path=None):
        """
        TODO: Fill out when complete
        :param config_path: Location where a configuration .csv is located
        :param model_checkpoint: Specific model checkpoint to load
        :param data_path: Location for all data to be inferenced. Only reference the directory.
        """
        self.cfg = ModelConfig()
        self.cfg.load(config_path)

        # Initialize the data handler
        super().__init__(self.cfg)

        self.model = None
        self.model_graph = tf.Graph()
        self.model_inferencing_files = []

        self._model_checkpoint_path = model_checkpoint
        self._inference_data_path = data_path

    def setup(self):
        # Attempt to pull in a saved model. This may take a bit depending on size.
        print(bcolors.OKBLUE + 'Initializing: ' + self.cfg.model_name + bcolors.ENDC)
        print(bcolors.OKBLUE + 'This may take a bit, so please be patient' + bcolors.ENDC)
        try:
            with self.model_graph.as_default(), tf.variable_scope(self.cfg.variable_scope):
                self.model = self._generate_model()
                self.model.load(self._model_checkpoint_path)
        except:
            print("Model failed loading. Check the config settings vs the model you think you are loading.")
            raise Exception

        print(bcolors.OKBLUE + 'Initialization Success' + bcolors.ENDC)

        # Figure out what files are available for inferencing
        self.model_inferencing_files = glob.glob(self._inference_data_path + '*.csv')

    def predict(self, file=None, existing_model=None):
        """
        Predicts the output of a model given a set of inputs from a file.
        If an existing model is to be used, setup() does not need to be called
        :param file:
        :param existing_model:
        :return: Numpy array of output data
        """
        x, actual_y = super().generate_validation_data(file)

        if existing_model is None:
            predict_y = self.model.predict(x)
            predict_y = self.model.predict(x)
        else:
            predict_y = existing_model.predict(x)

        return predict_y, actual_y

    def evaluate(self, file=None, existing_model=None):
        x, y = super().generate_validation_data(file)

        # The session is needed to know what state the model was in. Apparently,
        # the evaluate function does not work without it.
        if existing_model is None:
            with self.model.session:
                score = self.model.evaluate(x, y)
        else:
            with existing_model.session:
                score = existing_model.evaluate(x, y)

        return score

    def _generate_model(self):
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

        # All models are guaranteed to have this input form. No checkpoint paths
        # are specified because this model will never be used for training.
        return model_func(shape=model_input_shape,
                          dim_in=self.cfg.input_size,
                          dim_out=self.cfg.output_size,
                          past_depth=self.cfg.input_depth,
                          layer_neurons=self.cfg.neurons_per_layer,
                          layer_dropout=self.cfg.layer_dropout,
                          learning_rate=self.cfg.learning_rate)


class DroneModel:
    available_euler_models = ['pitch', 'roll', 'yaw', 'full_euler_model']
    available_gyro_models = ['gx', 'gy', 'gz', 'full_gyro_model']

    def __init__(self, euler_cfg_dict={}, euler_mdl_dict={}, gyro_cfg_dict={}, gyro_mdl_dict={}):
        """
        Expects the user to pass in a variable number of paths to the kinds of models implemented.
        To correctly pass in the paths, use the available_euler/gyro_models lists as keys.

        :param euler_cfg_dict: Dictionary of paths to 'pitch', 'roll', 'yaw', or 'full_model' model configs
        :param gyro_cfg_dict: Dictionary of paths to 'gx', 'gy', 'gz', or 'full_model' model configs
        """
        for key in euler_cfg_dict:
            if key not in self.available_euler_models:
                print(bcolors.WARNING + 'Invalid key found in euler_cfg_dict!' + bcolors.ENDC)
                raise ValueError

            if key not in euler_mdl_dict:
                print(bcolors.WARNING + 'Key not found in euler_mdl_dict!' + bcolors.ENDC)
                raise ValueError

        for key in gyro_cfg_dict:
            if key not in self.available_gyro_models:
                print(bcolors.WARNING + 'Invalid key found in gyro_cfg_dict!' + bcolors.ENDC)
                raise ValueError

            if key not in gyro_mdl_dict:
                print(bcolors.WARNING + 'Key not found in gyro_mdl_dict!' + bcolors.ENDC)
                raise ValueError

        # --------------------------------
        # Euler Model Setup
        # --------------------------------
        # Load in all given Euler Model configuration files if key is valid
        self._euler_cfg = {key: ModelConfig() for key in euler_cfg_dict}
        [self._euler_cfg[key].load(euler_cfg_dict[key]) for key in self._euler_cfg]

        # Generate the model placeholders & checkpoint locations
        self._euler_model = {key: None for key in self._euler_cfg}
        self._euler_graph = {key: tf.Graph() for key in self._euler_cfg}
        self._euler_chkpt = euler_mdl_dict

        # Create the training/validation data handlers. This only initializes
        # valid keys found from the input dict
        self._euler_data = {key: DataHandler(self._euler_cfg[key]) for key in self._euler_cfg}

        # --------------------------------
        # Gyro Model Setup
        # --------------------------------
        self._gyro_cfg = {key: ModelConfig() for key in gyro_cfg_dict}
        [self._gyro_cfg[key].load(gyro_cfg_dict[key]) for key in self._gyro_cfg]

        self._gyro_model = {key: None for key in self._gyro_cfg}
        self._gyro_graph = {key: tf.Graph() for key in self._gyro_cfg}
        self._gyro_chkpt = gyro_mdl_dict

        self._gyro_data = {key: DataHandler(self._gyro_cfg[key]) for key in self._gyro_cfg}


        self._matlab_engine = None

        self._pitch_ctrl = None
        self._roll_ctrl = None
        self._yaw_ctrl = None

        self._pid_sample_time_mS = 8.0  # Valkyrie FCS pid controller update rate
        self._sim_dt = 0.002

        # TODO: Update these externally somehow
        self.gyro_symmetric_range_actual = Converter.gyro_symmetric_range     # The actual +/- data range recorded by the gyro
        self.gyro_symmetric_range_mapped = Converter.mapped_gyro_symmetric_range       # The desired +/- data range input for the NN
        self.motor_range_actual_max = Converter.motor_max          # ESC input max throttle signal in mS
        self.motor_range_actual_min = Converter.motor_min             # ESC input min throttle signal in mS
        self.motor_range_mapped_max = Converter.mapped_motor_max             # NN input max throttle signal (unitless)
        self.motor_range_mapped_min = Converter.mapped_motor_min             # NN input min throttle signal (unitless)
        self.ahrs_range_actual_min = Converter.ahrs_min
        self.ahrs_range_actual_max = Converter.ahrs_max
        self.ahrs_range_mapped_min = Converter.mapped_ahrs_min
        self.ahrs_range_mapped_max = Converter.mapped_ahrs_max

    def initialize(self, ahrs_sample_freq, pid_update_freq):
        self._pid_sample_time_mS = int((1.0 / pid_update_freq) * 1000.0)
        self._sim_dt = 1.0 / ahrs_sample_freq

        # -----------------------------
        # Initialize the Euler Models
        # -----------------------------
        for key in self._euler_cfg:
            with self._euler_graph[key].as_default(), tf.variable_scope(self._euler_cfg[key].variable_scope):
                print(bcolors.OKGREEN + 'Initializing Euler Model: ' + key + bcolors.ENDC)
                print(bcolors.OKGREEN + 'Model Name: ' + self._euler_cfg[key].model_name + bcolors.ENDC)

                self._euler_model[key] = self._generate_model(config=self._euler_cfg[key])
                self._euler_model[key].load(self._euler_chkpt[key])

                print(bcolors.OKGREEN + 'Loaded archived model' + bcolors.ENDC + '\n')

        # -----------------------------
        # Initialize the Gyro Models
        # -----------------------------
        for key in self._gyro_cfg:
            with self._gyro_graph[key].as_default(), tf.variable_scope(self._gyro_cfg[key].variable_scope):
                print(bcolors.OKGREEN + 'Initializing Gyro Model: ' + key + bcolors.ENDC)
                print(bcolors.OKGREEN + 'Model Name: ' + self._gyro_cfg[key].model_name + bcolors.ENDC)

                self._gyro_model[key] = self._generate_model(config=self._gyro_cfg[key])
                self._gyro_model[key].load(self._gyro_chkpt[key])

                print(bcolors.OKGREEN + 'Loaded archived model' + bcolors.ENDC + '\n')

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
                                          angle_direction=AxisController.REVERSE,
                                          rate_direction=AxisController.REVERSE,
                                          sample_time_ms=self._pid_sample_time_mS)

        self._roll_ctrl = AxisController(angular_rate_range=100.0,
                                         motor_cmd_range=500.0,
                                         angle_direction=AxisController.DIRECT,
                                         rate_direction=AxisController.DIRECT,
                                         sample_time_ms=self._pid_sample_time_mS)

        self._yaw_ctrl = AxisController(angular_rate_range=100.0,
                                        motor_cmd_range=500.0,
                                        angle_direction=True,
                                        rate_direction=True,
                                        sample_time_ms=self._pid_sample_time_mS)

    def _simulate_axis(self, axis='', rate_axis='', input_sig=None, t_start=0.0, t_end=10.0,
                       enable_rate_control=False, invert_rate_direction=False):
        """
        Simulates a particular axis's response to a given input signal over the range of t_start to t_end.
        By default, only angle PID control is utilized. It is assumed that all PID constants have been
        set to appropriate values before calling this method.

        :param axis: Requested axis to simulate; valid values are 'pitch' 'roll' and 'yaw'
        :param rate_axis: Axis around which param 'axis' rotates; valid values are 'gx', 'gy', 'gz'
        :param input_sig: Dynamic signal that acts as input to the angle PID controller
        :param t_start: simulation start time
        :param t_end: simulation end time
        :param enable_rate_control: Turns on or off rate control. Default False
        :param invert_rate_direction: multiplies the predicted gyro signal by -1.0
        :return:
        """
        # -----------------------------------
        # Run a few precursory checks
        # -----------------------------------
        # Make sure the angle predictor model exists
        if axis not in self._euler_model:
            raise ValueError(bcolors.FAIL +
                             'Requested axis ' + axis + ' does not exist in initialized models.' +
                             bcolors.ENDC)

        # Make sure the angular rate predictor model exists
        if enable_rate_control:
            if rate_axis not in self._gyro_cfg:
                raise ValueError(bcolors.FAIL +
                                 'Requested rate axis ' + rate_axis + ' does not exist in initialized models.' +
                                 bcolors.ENDC)

        # Force [1xN] input vector where N is the total simulation steps
        in_shape = np.shape(input_sig)
        assert(in_shape[0] == 1)
        assert(in_shape[1] >= int(np.ceil((t_end - t_start) / self._sim_dt)))

        # Both euler and gyro models should have been trained on the same shape input
        if self._euler_cfg[axis].data_inversion:
            euler_input_shape = [None, self._euler_cfg[axis].input_size, self._euler_cfg[axis].input_depth]
        else:
            euler_input_shape = [None, self._euler_cfg[axis].input_depth, self._euler_cfg[axis].input_size]

        if enable_rate_control:
            if self._gyro_cfg[rate_axis].data_inversion:
                gyro_input_shape = [None, self._gyro_cfg[rate_axis].input_size, self._gyro_cfg[rate_axis].input_depth]
            else:
                gyro_input_shape = [None, self._gyro_cfg[rate_axis].input_depth, self._gyro_cfg[rate_axis].input_size]

            assert(euler_input_shape == gyro_input_shape)
            del gyro_input_shape

        input_shape = euler_input_shape
        del euler_input_shape

        # -----------------------------------
        # Initialize simulation vars
        # -----------------------------------
        num_sim_steps = int(np.ceil((t_end - t_start) / self._sim_dt))
        time_history = np.linspace(t_start, t_end, num_sim_steps)

        base_throttle = 1160.0
        pitch_cmd = 0
        roll_cmd = 0
        yaw_cmd = 0

        # Serves as input to both models. Latest value is in the last column [:, -1]
        model_input_series = np.zeros([input_shape[1], input_shape[2]])

        # Logs output of models
        euler_output = np.zeros([1, num_sim_steps])
        gyro_output = np.zeros([1, num_sim_steps])

        # Logs the most recent input to the neural network (does not include full time history)
        model_cmd_log = np.zeros([self._euler_cfg[axis].input_size, num_sim_steps])
        in_sig_cmd_log = np.zeros([1, num_sim_steps])

        # -----------------------------------
        # Run the full simulation of the input signals
        # -----------------------------------
        print(bcolors.OKGREEN + 'Starting simulation of ' + axis + ' axis' + bcolors.ENDC)
        tick_start = time.perf_counter()

        current_step = 0
        current_time = 0.0
        last_time = 0.0
        while current_step < num_sim_steps:

            if current_step % 100 == 0:
                print(bcolors.OKGREEN + 'Step ' + str(current_step) + ' of ' + str(num_sim_steps) + bcolors.ENDC)

            # -----------------------------------
            # Generate a new motor signal
            # -----------------------------------
            in_sig_cmd_log[0, current_step] = input_sig[0, current_step]

            # Only update the pid controller at the correct frequency
            if (current_time - last_time) > (self._pid_sample_time_mS / 1000.0):
                last_time = current_time

                if axis == 'pitch':
                    pitch_cmd = self._step_pitch_controller(
                        angle_setpoint=input_sig[0, current_step],          # Current angle setpoint
                        angle_feedback=euler_output[0, current_step-1],     # Last computed pitch angle
                        rate_feedback=gyro_output[0, current_step-1],       # Last computed pitch rate
                        en_rate_ctrl=enable_rate_control)

                elif axis == 'roll':
                    roll_cmd = self._step_roll_controller(
                        angle_setpoint=input_sig[0, current_step],          # Current angle setpoint
                        angle_feedback=euler_output[0, current_step - 1],   # Last computed pitch angle
                        rate_feedback=gyro_output[0, current_step - 1],     # Last computed pitch rate
                        en_rate_ctrl=enable_rate_control)

                elif axis == 'yaw':
                    yaw_cmd = self._step_yaw_controller(
                        angle_setpoint=input_sig[0, current_step],          # Current angle setpoint
                        angle_feedback=euler_output[0, current_step-1],     # Last computed yaw angle
                        rate_feedback=gyro_output[0, current_step-1],       # Last computed yaw rate
                        en_rate_ctrl=enable_rate_control)

            # Add some noise to help excite internal mode dynamics
            noise = np.random.uniform(0.0, 1.0, [self._euler_cfg[axis].input_size, 1])
            motor_signal = noise + MotorController.generate_motor_signals(base_throttle, pitch_cmd, roll_cmd, yaw_cmd)

            # -----------------------------------
            # Generate a new NN input
            # -----------------------------------
            # Roll the time history backwards so that the oldest value pops up at the latest index

            if input_shape[1] > input_shape[2]: # [time steps x input dim]
                model_input_series = np.roll(model_input_series, -1, axis=0)
                model_input_series[-1, :] = motor_signal[:, 0].transpose()

            else:  # [input dim x time steps]
                model_input_series = np.roll(model_input_series, -1, axis=1)
                model_input_series[:, -1] = motor_signal[:, 0]

            # Map the input data into the range expected by the NN
            mapped_model_input_series = self._real_motor_data_to_mapped(model_input_series)
            model_cmd_log[:, current_step] = self._real_motor_data_to_mapped(motor_signal)[:, 0]

            # -----------------------------------
            # Predict new Euler/Gyro outputs
            # -----------------------------------
            # Expand so that the 0th axis tells the NN what our batch size is: aka 1.
            model_input = np.expand_dims(mapped_model_input_series, axis=0)

            euler_prediction = self._euler_model[axis].predict(model_input)
            euler_output[0, current_step] = self._mapped_ahrs_data_to_real(euler_prediction[0])

            if enable_rate_control:
                gyro_prediction = self._gyro_model[rate_axis].predict(model_input)
                gyro_output[0, current_step] = self._mapped_gyro_data_to_real(gyro_prediction[0])

                if invert_rate_direction:
                    gyro_output[0, current_step] *= -1.0

            # Update sim vars
            current_step += 1
            current_time += self._sim_dt

        tick_end = time.perf_counter()
        elapsed_time = tick_end - tick_start
        print("Total Time: ", elapsed_time)

        results = {'euler_output': euler_output,
                   'gyro_output': gyro_output,
                   'model_input': model_cmd_log,
                   'signal_input': in_sig_cmd_log,
                   'time': time_history}

        return results

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

    def analyze_step_performance_siso(self, x, y, expected_final_value):
        assert(x.ndim == 1)
        assert(np.isscalar(expected_final_value))

        input_mdt = numpy_matrix_to_matlab(x)
        time_mdt = numpy_matrix_to_matlab(y)
        return self._matlab_engine.CalculateStepPerformance(input_mdt, time_mdt, expected_final_value)

    def TESTFUNC(self):
        path = 'C:\\Users\\Valkyrie\\Desktop\\TEMP\\'
        file = 'sim_cmd.json'

        filename = path + file
        last_m_time = 0

        # Loop simulations until manually stopped.
        while True:

            # Wait until the C++ code executes and produces a sim command file
            while True:
                if not os.path.exists(filename):
                    print(bcolors.OKBLUE + 'File not found yet...' + bcolors.ENDC)
                    time.sleep(1)

                elif os.path.getmtime(filename) != last_m_time:
                    last_m_time = os.path.getmtime(filename)

                    # Wait until the file is unlocked for safe access
                    while is_locked(filename):
                        time.sleep(1)

                    print(bcolors.OKGREEN + 'File successfully opened! Here is the data:' + bcolors.ENDC)
                    with open(filename) as jdat:
                        sim_data = json.load(jdat)

                    pprint(sim_data)
                    break

            # Pull out the json data
            axis = sim_data['axis']
            step_magnitude = sim_data['stepMagnitude']
            start_time = sim_data['startTime']
            end_time = sim_data['endTime']
            pid_vals = sim_data['pid']
            num_time_steps = sim_data['numTimeSteps']

            test_sig = np.zeros([1, num_time_steps])
            test_sig[0, 500:num_time_steps] = step_magnitude

            if axis == 'pitch':
                self.set_pitch_ctrl_pid(pid_vals['kp'], pid_vals['ki'], pid_vals['kd'], 0.9, 2.5, 0.01)
            elif axis == 'roll':
                self.set_roll_ctrl_pid(pid_vals['kp'], pid_vals['ki'], pid_vals['kd'], 0.9, 2.5, 0.01)
            elif axis == 'yaw':
                self.set_yaw_ctrl_pid(pid_vals['kp'], pid_vals['ki'], pid_vals['kd'], 0.9, 2.5, 0.01)

            # Simulate
            sim_out = self._simulate_axis(axis=axis, input_sig=test_sig, t_start=start_time, t_end=end_time)

            step_analysis = self.analyze_step_performance_siso(x=sim_out['time'],
                                                               y=sim_out['euler_output'],
                                                               expected_final_value=step_magnitude)

            # Return some fake results to the c++ code
            full_result = {'riseTime': step_analysis['RiseTime'],
                           'settlingTime': step_analysis['SettlingTime'],
                           'settlingMin': step_analysis['SettlingMin'],
                           'settlingMax': step_analysis['SettlingMax'],
                           'overshoot': step_analysis['Overshoot'],
                           'undershoot': step_analysis['Undershoot'],
                           'peak': step_analysis['Peak'],
                           'peakTime': step_analysis['PeakTime'],
                           'steadyStateError': sim_out['euler_output'][0, -1] - step_magnitude}

            for key, value in full_result.items():
                if np.isnan(value):
                    full_result[key] = -1.0

            with open(path + 'sim_result.json', 'w') as outfile:
                json.dump(full_result, outfile)



        # lw = 0.8
        # plt.figure(figsize=(32, 18))
        # plt.suptitle('Pitch Angle Step Test')
        # plt.plot(results['time'], results['euler_output'][0, :], 'g-', label='Output', linewidth=lw)
        # plt.plot(results['time'], results['signal_input'][0, :], 'b-', label='Input', linewidth=lw)
        # plt.legend()
        # plt.show()

    def TESTFUNC2(self):

        # Pull out the json data
        axis            = 'pitch'
        rate_axis       = 'gy'
        step_magnitude  = 0.0
        start_time      = 0.0
        end_time        = 3.9
        num_time_steps  = 2000

        test_sig = np.zeros([1, num_time_steps])
        test_sig[0, 500:num_time_steps] = step_magnitude

        if axis == 'pitch':
            self.set_pitch_ctrl_pid(3.5, 1.0, 0.01, 0.9, 2.5, 0.01)
        elif axis == 'roll':
            self.set_roll_ctrl_pid(2.4, 1.5, 1.1, 0.9, 2.5, 0.01)
        elif axis == 'yaw':
            self.set_yaw_ctrl_pid(2.4, 1.5, 1.1, 0.9, 2.5, 0.01)

        # Simulate
        sim_out = self._simulate_axis(axis=axis, rate_axis=rate_axis, input_sig=test_sig,
                                      t_start=start_time, t_end=end_time,
                                      enable_rate_control=False,
                                      invert_rate_direction=False)

        step_analysis = self.analyze_step_performance_siso(x=sim_out['time'],
                                                           y=sim_out['euler_output'],
                                                           expected_final_value=step_magnitude)

        print(step_analysis)

        lw = 0.8
        plt.figure(figsize=(32, 18))
        plt.suptitle('Pitch Angle Step Test')
        plt.plot(sim_out['time'], sim_out['euler_output'][0, :], 'g-', label='Angle Output', linewidth=lw)
        plt.plot(sim_out['time'], sim_out['gyro_output'][0, :], 'k-', label='Gyro Output', linewidth=lw)
        plt.plot(sim_out['time'], sim_out['signal_input'][0, :], 'b-', label='Angle Input', linewidth=lw)
        plt.legend()

        plt.figure(figsize=(32, 18))
        plt.suptitle('Pitch Angle Step Test')
        plt.plot(sim_out['time'], sim_out['model_input'][0, :], 'g-', label='m1CMD', linewidth=lw)
        plt.plot(sim_out['time'], sim_out['model_input'][1, :], 'k-', label='m2CMD', linewidth=lw)
        plt.plot(sim_out['time'], sim_out['model_input'][2, :], 'b-', label='m3CMD', linewidth=lw)
        plt.plot(sim_out['time'], sim_out['model_input'][3, :], 'c-', label='m4CMD', linewidth=lw)
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

    # ---------------------------------------------
    # Private Methods
    # ---------------------------------------------
    def _generate_model(self, config=None):
        """
        A factory function to generate a system model based upon config object
        """
        if config.data_inversion:
            model_input_shape = [None, config.input_size, config.input_depth]
        else:
            model_input_shape = [None, config.input_depth, config.input_size]

        # Grab the particular model function in use
        assert (isinstance(config.model_type, str))
        if config.model_type in TFModels.function_dispatcher:
            model_func = TFModels.function_dispatcher[config.model_type]
        else:
            raise ValueError(bcolors.FAIL + 'Invalid model type!' + bcolors.ENDC)

        # All models are guaranteed to have this input form. No checkpoint paths
        # are specified because this model will never be used for training.
        return model_func(shape=model_input_shape,
                          dim_in=config.input_size,
                          dim_out=config.output_size,
                          past_depth=config.input_depth,
                          layer_neurons=config.neurons_per_layer,
                          layer_dropout=config.layer_dropout,
                          learning_rate=config.learning_rate)

    def _predict_model(self, model_key=None, input_sig=None):
        if model_key in self._euler_model:
            return self._euler_model[model_key].predict(input_sig)
        elif model_key in self._gyro_model:
            return self._gyro_model[model_key].predict(input_sig)

    def _smooth_raw_output(self):
        """
        Implements a smoothing filter so that we don't have so much noise to
        process
        :return:
        """
        raise NotImplementedError

    def _parse_model_config(self):
        raise NotImplementedError

    def _step_pitch_controller(self, angle_setpoint, angle_feedback, rate_feedback, en_rate_ctrl=False):
        self._pitch_ctrl.angle_setpoint = angle_setpoint
        self._pitch_ctrl.angle_feedback = angle_feedback
        self._pitch_ctrl.rate_feedback = rate_feedback

        self._pitch_ctrl.compute(en_rate_ctrl)

        return self._pitch_ctrl.controller_output

    def _step_roll_controller(self, angle_setpoint, angle_feedback, rate_feedback, en_rate_ctrl=False):
        self._roll_ctrl.angle_setpoint = angle_setpoint
        self._roll_ctrl.angle_feedback = angle_feedback
        self._roll_ctrl.rate_feedback = rate_feedback

        self._roll_ctrl.compute(en_rate_ctrl)

        return self._roll_ctrl.controller_output

    def _step_yaw_controller(self, angle_setpoint, angle_feedback, rate_feedback, en_rate_ctrl=False):
        self._yaw_ctrl.angle_setpoint = angle_setpoint
        self._yaw_ctrl.angle_feedback = angle_feedback
        self._yaw_ctrl.rate_feedback = rate_feedback

        self._yaw_ctrl.compute(en_rate_ctrl)

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

    def _real_ahrs_data_to_mapped(self, input_signal):
        return np.interp(input_signal,
                         [self.ahrs_range_actual_min, self.ahrs_range_actual_max],
                         [self.ahrs_range_mapped_min, self.ahrs_range_mapped_max])

    def _mapped_ahrs_data_to_real(self, input_signal):
        return np.interp(input_signal,
                         [self.ahrs_range_mapped_min, self.ahrs_range_mapped_max],
                         [self.ahrs_range_actual_min, self.ahrs_range_actual_max])


if __name__ == "__main__":
    predict_off_file = True
    predict_off_sim = False


    cfg = 'G:/Projects/ValkyrieRNN/Simulation/SmallInferenceTestModel/config.csv'
    ckpt = 'G:/Projects/ValkyrieRNN/Simulation/SmallInferenceTestModel/training/best_results/SmallInferenceTestModel7816'

    # cfg = 'G:/Projects/ValkyrieRNN/Simulation/pitch_full_ver1/config.csv'
    # ckpt = 'G:/Projects/ValkyrieRNN/Simulation/pitch_full_ver1/training/best_results/pitch_full_ver1592726'

    cfg = 'G:/Projects/ValkyrieRNN/Simulation/pitch_full_ver2/config.csv'
    ckpt = 'G:/Projects/ValkyrieRNN/Simulation/pitch_full_ver2/training/best_results/pitch_full_ver2218939'

    gy_cfg = 'G:/Projects/ValkyrieRNN/Simulation/gyro_y_full_ver2/config.csv'
    gy_ckpt = 'G:/Projects/ValkyrieRNN/Simulation/gyro_y_full_ver2/training/best_results/gyro_y_full_ver282006'

    data = 'G:/Projects/ValkyrieRNN/Data/ValidationData/'

    if predict_off_sim:
        model = DroneModel(euler_cfg_dict={'pitch': cfg},
                           euler_mdl_dict={'pitch': ckpt},
                           gyro_cfg_dict={'gy': gy_cfg},
                           gyro_mdl_dict={'gy': gy_ckpt})
        model.initialize(500, 125)
        model.TESTFUNC2()

    if predict_off_file:
        model = ModelInferencer(config_path=cfg, model_checkpoint=ckpt, data_path=data)
        model.setup()

        files = model.model_inferencing_files

        ypredict, yactual = model.predict(file=files[5])

        lw = 0.8
        plt.figure(figsize=(32, 18))
        plt.suptitle('Pitch Angle Step Test')
        plt.plot(ypredict[:, 0], 'g-', label='Prediction', linewidth=lw)
        plt.plot(yactual[:, 0], 'b-', label='Actual', linewidth=lw)
        plt.legend()
        plt.show()




