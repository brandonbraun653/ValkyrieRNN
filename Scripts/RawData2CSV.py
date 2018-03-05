from Scripts import DataStructures
from src.Miscellaneous import bcolors
from collections import OrderedDict

import os
import pandas as pd
import numpy as np
import matlab.engine

matlab_smoothTimeSeriesFile = "SmoothTimeSeriesLog.m"

input_ahrsLogFile_minimal   = "DroneData/raw/ahrsLogMinimal.dat"
input_ahrsLogFile_full      = "DroneData/raw/ahrsLogFull.dat"
input_motorLogFile          = "DroneData/raw/motorLog.dat"
input_angleSetpointLogFile  = "DroneData/raw/angleSetpoints.dat"
input_rateSetpointLogFile   = "DroneData/raw/rateSetpoints.dat"

output_ahrsCSVFile          = "DroneData/csv/ahrsLog.csv"
output_motorCSVFile         = "DroneData/csv/motorLog.csv"
output_angleSetpointCSVFile = "DroneData/csv/angleSetpoints.csv"
output_rateSetpointCSVFile  = "DroneData/csv/rateSetpoints.csv"
output_TSBlankSpaceCSVFile  = "DroneData/csv/timeSeriesDataRaw.csv"
output_TSInterpCSVFIle      = "DroneData/csv/timeSeriesDataInterpolated.csv"

header_timeSeriesCSV = "rtosTick,pitch,roll,yaw," + \
                       "ax,ay,az,gx,gy,gz,mx,my,mz," +\
                       "m1CMD,m2CMD,m3CMD,m4CMD," +\
                       "asp,asr,asy,rsp,rsr,rsy\n"


class DataConverter:
    def __init__(self, base_path, raw_path, csv_path, matlab_path):
        """
        Instruct the conversion module where to find all the relevant information
        :param base_path: Root directory for all data
        :param raw_path: Directory containing .dat log files
        :param csv_path: Directory for placing csv results
        :param matlab_path: Directory containing matlab scripts
        """
        self._base_path = base_path
        self._raw_path = raw_path
        self._csv_path = csv_path
        self._matlab_path = matlab_path

        self._raw_dat_filenames = []
        self.data_folders = []

    def add_dat_filenames(self, filenames):
        self._raw_dat_filenames = filenames

    def add_dat_folders(self, folders):
        self.data_folders = folders

    def raw_data_2_csv(self):
        """
        Takes the raw byte code from the SD Card logs and converts them into easy to
        interpret CSV files
        """
        for folder in self.data_folders:
            print(bcolors.OKGREEN + 'Changing directory to: ' + self._raw_path+folder + bcolors.ENDC)
            for file in self._raw_dat_filenames:
                # ----------------------
                # Parse the AHRS data
                # ----------------------
                if 'ahrsLog' in file:
                    try:
                        input_ahrs_log_file = self._raw_path + folder + file
                        output_ahrs_csv_file = self._csv_path + folder + 'ahrsLog.csv'

                        if not os.path.exists(self._csv_path + folder):
                            os.makedirs(self._csv_path + folder)

                        raw_ahrs_data = []
                        input_file_size = os.path.getsize(input_ahrs_log_file)

                        with open(input_ahrs_log_file, "rb") as input_file:
                            bytes_read = 0

                            while bytes_read < input_file_size:
                                if 'Full' in file:
                                    measurement = DataStructures.SDLogAHRSFull()
                                elif 'Minimal' in file:
                                    measurement = DataStructures.SDLogAHRSMinimal()

                                measurement.unpack_raw_hex(input_file.read(measurement.structSizeInBytes))
                                raw_ahrs_data.append(measurement)
                                bytes_read += measurement.structSizeInBytes

                            # Write all the measurements to a csv file for processing later
                            with open(output_ahrs_csv_file, 'w') as output_file:
                                [output_file.write(raw_ahrs_data[x].as_csv()) for x in range(len(raw_ahrs_data))]

                        print("Done parsing AHRS log file.")

                    except:
                        print("AHRS file doesn't exist. Continuing on.")

                # ----------------------
                # Parse the motor data
                # ----------------------
                if 'motor' in file:
                    try:
                        input_motor_log_file = self._raw_path + folder + file
                        output_motor_csv_file = self._csv_path + folder + 'motorLog.csv'

                        if not os.path.exists(self._csv_path + folder):
                            os.makedirs(self._csv_path + folder)

                        raw_motor_data = []
                        input_file_size = os.path.getsize(input_motor_log_file)

                        with open(input_motor_log_file, "rb") as input_file:
                            bytes_read = 0

                            while bytes_read < input_file_size:
                                measurement = DataStructures.SDLogMotor()
                                measurement.unpack_raw_hex(input_file.read(measurement.structSizeInBytes))
                                raw_motor_data.append(measurement)
                                bytes_read += measurement.structSizeInBytes

                            # Write all the measurements to a csv file for processing later
                            with open(output_motor_csv_file, 'w') as output_file:
                                [output_file.write(raw_motor_data[x].as_csv()) for x in range(len(raw_motor_data))]

                        print("Done parsing Motor log file.")

                    except:
                        print("Motor file doesn't exist. Continuing on.")

                # ----------------------
                # Parse the angle setpoint data
                # ----------------------
                if 'angle' in file:
                    try:
                        input_angle_setpoint_log_file = self._raw_path + folder + file
                        output_angle_setpoint_csv_file = self._csv_path + folder + 'angleSetpoints.csv'

                        if not os.path.exists(self._csv_path + folder):
                            os.makedirs(self._csv_path + folder)

                        raw_angle_setpoint_data = []
                        input_file_size = os.path.getsize(input_angle_setpoint_log_file)

                        with open(input_angle_setpoint_log_file, "rb") as input_file:
                            bytes_read = 0

                            while bytes_read < input_file_size:
                                measurement = DataStructures.SDLogAngleSetpoint()
                                measurement.unpack_raw_hex(input_file.read(measurement.structSizeInBytes))

                                raw_angle_setpoint_data.append(measurement)

                                bytes_read += measurement.structSizeInBytes

                            with open(output_angle_setpoint_csv_file, 'w') as output_file:
                                [output_file.write(raw_angle_setpoint_data[x].as_csv()) for x in range(len(raw_angle_setpoint_data))]

                        print("Done parsing Angle Setpoint log file.")

                    except:
                        print("Angle setpoint file doesn't exist. Continuing on.")

                # ----------------------
                # Parse the rate setpoint data
                # ----------------------
                if 'rate' in file:
                    try:
                        input_rate_setpoint_log_file = self._raw_path + folder + file
                        output_rate_setpoint_csv_file = self._csv_path + folder + 'rateSetpoints.csv'

                        if not os.path.exists(self._csv_path + folder):
                            os.makedirs(self._csv_path + folder)

                        raw_rate_setpoint_data = []
                        input_file_size = os.path.getsize(input_rate_setpoint_log_file)

                        with open(input_rate_setpoint_log_file, "rb") as input_file:
                            bytes_read = 0

                            while bytes_read < input_file_size:
                                measurement = DataStructures.SDLogRateSetpoint()
                                measurement.unpack_raw_hex(input_file.read(measurement.structSizeInBytes))

                                raw_rate_setpoint_data.append(measurement)

                                bytes_read += measurement.structSizeInBytes

                            with open(output_rate_setpoint_csv_file, 'w') as output_file:
                                [output_file.write(raw_rate_setpoint_data[x].as_csv()) for x in range(len(raw_rate_setpoint_data))]

                        print("Done parsing Rate Setpoint log file.")

                    except:
                        print("Rate setpoint file doesn't exist. Continuing on.")

    def create_time_series_from_csv_logs(self, ahrs_type_full=True):
        """
        This function takes the output of raw_data_2_csv and filters through it to produce a single csv file with
        concatenated sensor data at each time step. Because the freeRTOS implementation in ValkyrieFCS has slight delays
        between tasks, there are "gaps" in the information that must be filled in.
        """
        def stringify(data):
            csv_stringified_data = [(str(x) + ',') for x in data]
            csv_stringified_data[-1] = csv_stringified_data[-1].replace(",", "")

            return ''.join(csv_stringified_data)

        def write_time_series_to_file(filename, header_str, data):
            data_keys = list(data.keys())
            data_keys.sort(key=int)

            with open(filename, 'w') as file:
                file.write(header_str)

                for key_val in data_keys:
                    line = str(key_val) + ',' + \
                           stringify(data[key_val]['ahrsMeas']) + ',' + \
                           stringify(data[key_val]['motorCMD']) + ',' + \
                           stringify(data[key_val]['angleSet']) + ',' + \
                           stringify(data[key_val]['rateSet']) + '\n'

                    file.write(line)

        print(bcolors.OKBLUE + "Starting creation of time series log file..." + bcolors.ENDC)
        for folder in self.data_folders:
            print(bcolors.OKGREEN + 'Changing directory to: ' + self._raw_path + folder + bcolors.ENDC)

            time_col = 0
            m1_col = 1
            m2_col = 2
            m3_col = 3
            m4_col = 4

            pitch_col = 1
            roll_col = 2
            yaw_col = 3
            ax_col = 4
            ay_col = 5
            az_col = 6
            gx_col = 7
            gy_col = 8
            gz_col = 9
            mx_col = 10
            my_col = 11
            mz_col = 12

            update_rate_ms = 8

            motor_csv_file          = self._csv_path + folder + 'motorLog.csv'
            ahrs_csv_file           = self._csv_path + folder + 'ahrsLog.csv'
            angle_setpoint_csv_file = self._csv_path + folder + 'angleSetpoints.csv'
            rate_setpoint_csv_file  = self._csv_path + folder + 'rateSetpoints.csv'

            raw_timeseries_csv_file = self._csv_path + folder + 'timeSeriesDataRaw.csv'
            int_timeseries_csv_file = self._csv_path + folder + 'timeSeriesDataInterpolated.csv'

            # ------------------------------
            # Grab every possible recorded freeRTOS tick value and
            # initialize the time_series_data dictionary to zero
            # ------------------------------
            #all_recorded_ticks = np.r_[motor_ticks, ahrs_ticks, angle_setpoint_ticks, rate_setpoint_ticks]
            all_recorded_ticks = np.empty([1, 1])

            motor_data_valid = True
            if os.path.exists(motor_csv_file):
                motor_data  = pd.read_csv(motor_csv_file, header=None)
                motor_ticks = np.array(motor_data[time_col]).reshape(-1, 1)

                all_recorded_ticks = np.r_[all_recorded_ticks, motor_ticks]

            else:
                motor_data_valid = False

            ahrs_data_valid = True
            if os.path.exists(ahrs_csv_file):
                ahrs_data = pd.read_csv(ahrs_csv_file, header=None)
                ahrs_ticks = np.array(ahrs_data[time_col]).reshape(-1, 1)

                all_recorded_ticks = np.r_[all_recorded_ticks, ahrs_ticks]

            else:
                ahrs_data_valid = False

            angle_setpoint_data_valid = True
            if os.path.exists(angle_setpoint_csv_file):
                angle_setpoint_data = pd.read_csv(angle_setpoint_csv_file, header=None)
                angle_setpoint_ticks = np.array(angle_setpoint_data[time_col]).reshape(-1, 1)

                all_recorded_ticks = np.r_[all_recorded_ticks, angle_setpoint_ticks]

            else:
                angle_setpoint_data_valid = False

            rate_setpoint_data_valid = True
            if os.path.exists(rate_setpoint_csv_file):
                rate_setpoint_data = pd.read_csv(rate_setpoint_csv_file, header=None)
                rate_setpoint_ticks = np.array(rate_setpoint_data[time_col]).reshape(-1, 1)

                all_recorded_ticks = np.r_[all_recorded_ticks, rate_setpoint_ticks]

            else:
                rate_setpoint_data_valid = False

            if all_recorded_ticks is None:
                print("No available data for this round. Continuing to next folder...")
                continue

            print("\tCreating time series structure...")
            time_series_data = {}
            for tic in all_recorded_ticks[:, 0]:

                if np.isnan(tic):
                    continue

                if int(tic) not in time_series_data:
                    time_series_data[int(tic)] = {
                        # [M1, M2, M3, M4]
                        'motorCMD': [0, 0, 0, 0],

                        # [pitch, roll, yaw, ax, ay, az, gx, gy, gz, mx, my, mz]
                        'ahrsMeas': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],

                        # [pitch, roll, yaw]
                        'angleSet': [0, 0, 0],

                        # [pitch, roll, yaw]
                        'rateSet':  [0, 0, 0]}

            # ------------------------------
            # Fill in the time_series_data dict with pertinent information
            # at each valid tic mark
            # ------------------------------
            if motor_data_valid:
                print("\tFilling motor data...")
                for idx in range(0, len(motor_ticks)):
                    tic = motor_data[time_col][idx]
                    time_series_data[tic]['motorCMD'] = np.array([motor_data[m1_col][idx],
                                                                  motor_data[m2_col][idx],
                                                                  motor_data[m3_col][idx],
                                                                  motor_data[m4_col][idx]])

            if ahrs_data_valid:
                print("\tFilling ahrs data...")
                for idx in range(0, len(ahrs_ticks)):
                    tic = ahrs_data[time_col][idx]

                    if ahrs_type_full:
                        time_series_data[tic]['ahrsMeas'] = np.array([
                            ahrs_data[pitch_col][idx], ahrs_data[roll_col][idx], ahrs_data[yaw_col][idx],
                            ahrs_data[ax_col][idx],    ahrs_data[ay_col][idx],   ahrs_data[az_col][idx],
                            ahrs_data[gx_col][idx],    ahrs_data[gy_col][idx],   ahrs_data[gz_col][idx],
                            ahrs_data[mx_col][idx],    ahrs_data[my_col][idx],   ahrs_data[mz_col][idx]])

                    else:  # Assumes minimal ahrs data logging type
                        time_series_data[tic]['ahrsMeas'] = np.array([
                            ahrs_data[pitch_col][idx], ahrs_data[roll_col][idx], ahrs_data[yaw_col][idx]])

            if angle_setpoint_data_valid:
                print("\tFilling angle setpoint data...")
                for idx in range(0, len(angle_setpoint_ticks)):
                    tic = angle_setpoint_data[time_col][idx]
                    time_series_data[tic]['angleSet'] = np.array([angle_setpoint_data[pitch_col][idx],
                                                                  angle_setpoint_data[roll_col][idx],
                                                                  angle_setpoint_data[yaw_col][idx]])

            if rate_setpoint_data_valid:
                print("\tFilling rate setpoint data...")
                for idx in range(0, len(rate_setpoint_ticks)):
                    tic = rate_setpoint_data[time_col][idx]
                    time_series_data[tic]['rateSet'] = np.array([rate_setpoint_data[pitch_col][idx],
                                                                 rate_setpoint_data[roll_col][idx],
                                                                 rate_setpoint_data[yaw_col][idx]])

            # As a sanity check for later, make sure the data is logged before interpolation
            print("\tWriting filled data to file")
            write_time_series_to_file(filename=raw_timeseries_csv_file,
                                      header_str=header_timeSeriesCSV,
                                      data=time_series_data)

            # ------------------------------
            # Linear interpolate the empty spaces left by ahrsData
            # ------------------------------
            if ahrs_data_valid:
                print("\tInterpolating AHRS...")
                keys = list(time_series_data.keys())
                keys.sort(key=int)

                for idx in range(0, len(keys)-1):
                    tic = keys[idx]

                    if np.sum(np.array(time_series_data[tic]['ahrsMeas'])) == 0:
                        current_idx = idx
                        num_tics = 1

                        # Find the next non zero tic
                        while time_series_data[keys[current_idx]]['ahrsMeas'][0] == 0.0:
                            num_tics += 1
                            current_idx += 1

                        # Generate the delta for each zero'd field
                        ahrs_prev = np.array(time_series_data[keys[idx - 1]]['ahrsMeas'])
                        ahrs_next = np.array(time_series_data[keys[current_idx]]['ahrsMeas'])

                        dVal = (ahrs_next - ahrs_prev) / num_tics

                        # Write the empty fields
                        for j in range(idx, current_idx):
                            new_value = np.array(time_series_data[keys[j - 1]]['ahrsMeas']) + dVal
                            time_series_data[keys[j]]['ahrsMeas'] = new_value

            # ------------------------------
            # Fill in the zero fields for motor commands (values don't change in MCU until rtos tick update)
            # ------------------------------
            if motor_data_valid:
                print("\tInterpolating Motor CMDs...")
                keys = list(time_series_data.keys())
                keys.sort(key=int)

                motors_on = False
                last_valid_cmd = []
                keys_to_remove = []
                for idx in range(0, len(keys)):
                    current_motor_cmd = np.array(time_series_data[keys[idx]]['motorCMD'])

                    if not motors_on:
                        # We don't want any data from before the motors turn on, so delete
                        # those lines from the final timeSeriesData
                        if current_motor_cmd[0] == 0:
                            keys_to_remove.append(keys[idx])
                        else:
                            motors_on = True
                            last_valid_cmd = current_motor_cmd
                            continue

                    if current_motor_cmd[0] == 0:
                        time_series_data[keys[idx]]['motorCMD'] = last_valid_cmd
                    else:
                        last_valid_cmd = current_motor_cmd

                for key in keys_to_remove:
                    time_series_data.pop(key)

            # ------------------------------
            # Fill in the zero fields for the PID controller
            # ------------------------------
            if angle_setpoint_data_valid or rate_setpoint_data_valid:
                print("\tInterpolating PID...")

            # Start with the angle controller for pitch, roll, yaw
            if angle_setpoint_data_valid:
                keys = list(time_series_data.keys())
                keys.sort(key=int)

                for controller in range(0, 3):
                    last_valid_cmd = 0
                    last_valid_key = 0

                    for idx in range(0, len(keys)):
                        current_key = keys[idx]
                        current_cmd = time_series_data[current_key]['angleSet'][controller]

                        dt = current_key - last_valid_key

                        # Not enough time has passed to change command
                        if dt % update_rate_ms == 0 and current_cmd != last_valid_cmd:
                            last_valid_key = current_key

                            # Check the next value...if they are the same, accept the input
                            if (current_key + update_rate_ms) in keys:
                                next1 = time_series_data[current_key + update_rate_ms]['angleSet'][controller]

                                if next1 == current_cmd:
                                    last_valid_cmd = current_cmd

                        time_series_data[keys[idx]]['angleSet'][controller] = last_valid_cmd

            # Then finish up with the rate controller
            if rate_setpoint_data_valid:
                keys = list(time_series_data.keys())
                keys.sort(key=int)

                for controller in range(0, 3):
                    last_valid_cmd = 0
                    last_valid_key = 0

                    for idx in range(0, len(keys)):
                        current_key = keys[idx]
                        current_cmd = time_series_data[current_key]['rateSet'][controller]

                        dt = current_key - last_valid_key

                        # Rate controller is far more volatile than the angle controller, so accept
                        # any change
                        if dt % update_rate_ms == 0 and current_cmd != last_valid_cmd:
                            last_valid_key = current_key
                            last_valid_cmd = current_cmd

                        time_series_data[keys[idx]]['rateSet'][controller] = last_valid_cmd

            # ------------------------------
            # Wrap up / Clean up
            # ------------------------------
            print("\tWriting results to file...")
            write_time_series_to_file(filename=int_timeseries_csv_file,
                                      header_str=header_timeSeriesCSV,
                                      data=time_series_data)

            print("Done")

    def smooth_time_series_data(self, gyro_max, gyro_min, motor_max, motor_min):
        """
        Uses several matlab specific commands to smooth out the time series data. Currently
        hardcoded for simplicity
        :return:
        """
        assert(np.isscalar(gyro_max))
        assert(np.isscalar(gyro_min))
        assert(np.isscalar(motor_max))
        assert(np.isscalar(motor_min))

        print("Starting Matlab Engine")
        eng = matlab.engine.start_matlab()
        eng.addpath(self._matlab_path, nargout=0)

        print("Executing data smoother")
        for folder in self.data_folders:
            print(bcolors.OKGREEN + 'Changing directory to: ' + self._raw_path + folder + bcolors.ENDC)
            eng.SmoothTimeSeriesData(self._csv_path + folder + 'timeSeriesDataInterpolated.csv',
                                     self._csv_path + folder,
                                     float(gyro_max),
                                     float(gyro_min),
                                     float(motor_max),
                                     float(motor_min),
                                     nargout=0)
        print("Done")


if __name__ == "__main__":

    parser = DataConverter(base_path='G:/Projects/ValkyrieRNN/TrainingData/',
                           raw_path='G:/Projects/ValkyrieRNN/TrainingData/raw/',
                           csv_path='G:/Projects/ValkyrieRNN/TrainingData/csv/',
                           matlab_path=r'C:/git/GitHub/ValkyrieRNN/Scripts/Matlab/')

    parser.add_dat_filenames(['ahrsLogFull.dat', 'angleSetpoints.dat', 'motorLog.dat', 'rateSetpoints.dat'])
    parser.add_dat_folders(['stepping_5_deg_coupled/',
                            'stepping_5_deg_uncoupled/',
                            'stepping_10_deg_coupled/',
                            'stepping_10_deg_uncoupled/',
                            'stepping_15_deg_coupled/',
                            'stepping_15_deg_uncoupled/',
                            'disturbances/'])

    parser.raw_data_2_csv()
    parser.create_time_series_from_csv_logs()

    gyro_max = 250
    gyro_min = -250
    motor_max = 1860
    motor_min = 1060
    parser.smooth_time_series_data(gyro_max=gyro_max,
                                   gyro_min=gyro_min,
                                   motor_max=motor_max,
                                   motor_min=motor_min)
