from Scripts import DataStructures

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
                       "asp,asr,asy,rsp,rsr,rsy\n" # angle_set_pitch, roll, yaw...rate_set_pitch, roll, yaw


def raw_data_2_csv(parseAHRS=True, parseMOTOR=True, parseANGLE=True, parseRATE=True, ahrs_type_full=True):
    """
    Takes the raw byte code from the SD Card logs and converts them into easy to
    interpret CSV files
    """

    # ----------------------
    # Parse the AHRS data
    # ----------------------
    if parseAHRS:
        try:
            if ahrs_type_full:
                input_ahrs_log_file = input_ahrsLogFile_full
            else:
                input_ahrs_log_file = input_ahrsLogFile_minimal

            raw_ahrs_data = []
            input_file_size = os.path.getsize(input_ahrs_log_file)

            with open(input_ahrs_log_file, "rb") as input_file:
                bytes_read = 0

                while bytes_read < input_file_size:
                    if ahrs_type_full:
                        measurement = DataStructures.SDLogAHRSFull()
                    else:
                        measurement = DataStructures.SDLogAHRSMinimal()

                    measurement.unpack_raw_hex(input_file.read(measurement.structSizeInBytes))

                    raw_ahrs_data.append(measurement)

                    bytes_read += measurement.structSizeInBytes

                # Write all the measurements to a csv file for processing later
                with open(output_ahrsCSVFile, 'w') as output_file:
                    [output_file.write(raw_ahrs_data[x].as_csv()) for x in range(len(raw_ahrs_data))]

            print("Done parsing AHRS log file.")

        except:
            print("Couldn't properly parse AHRS data. Continuing on.")

    # ----------------------
    # Parse the motor data
    # ----------------------
    if parseMOTOR:
        try:
            raw_motor_data = []
            input_file_size = os.path.getsize(input_motorLogFile)

            with open(input_motorLogFile, "rb") as input_file:
                bytes_read = 0

                while bytes_read < input_file_size:
                    measurement = DataStructures.SDLogMotor()
                    measurement.unpack_raw_hex(input_file.read(measurement.structSizeInBytes))

                    raw_motor_data.append(measurement)

                    bytes_read += measurement.structSizeInBytes

                # Write all the measurements to a csv file for processing later
                with open(output_motorCSVFile, 'w') as output_file:
                    [output_file.write(raw_motor_data[x].as_csv()) for x in range(len(raw_motor_data))]

            print("Done parsing Motor log file.")

        except:
            print("Couldn't properly parse motor data. Continuing on.")

    # ----------------------
    # Parse the angle setpoint data
    # ----------------------
    if parseANGLE:
        try:
            raw_angle_setpoint_data = []
            input_file_size = os.path.getsize(input_angleSetpointLogFile)

            with open(input_angleSetpointLogFile, "rb") as input_file:
                bytes_read = 0

                while bytes_read < input_file_size:
                    measurement = DataStructures.SDLogAngleSetpoint()
                    measurement.unpack_raw_hex(input_file.read(measurement.structSizeInBytes))

                    raw_angle_setpoint_data.append(measurement)

                    bytes_read += measurement.structSizeInBytes

                with open(output_angleSetpointCSVFile, 'w') as output_file:
                    [output_file.write(raw_angle_setpoint_data[x].as_csv()) for x in range(len(raw_angle_setpoint_data))]

            print("Done parsing Angle Setpoint log file.")

        except:
            print("Couldn't properly parse angle setpoint data. Continuing on.")

    # ----------------------
    # Parse the rate setpoint data
    # ----------------------
    if parseRATE:
        try:
            raw_rate_setpoint_data = []
            input_file_size = os.path.getsize(input_rateSetpointLogFile)

            with open(input_rateSetpointLogFile, "rb") as input_file:
                bytes_read = 0

                while bytes_read < input_file_size:
                    measurement = DataStructures.SDLogRateSetpoint()
                    measurement.unpack_raw_hex(input_file.read(measurement.structSizeInBytes))

                    raw_rate_setpoint_data.append(measurement)

                    bytes_read += measurement.structSizeInBytes

                with open(output_rateSetpointCSVFile, 'w') as output_file:
                    [output_file.write(raw_rate_setpoint_data[x].as_csv()) for x in range(len(raw_rate_setpoint_data))]

            print("Done parsing Rate Setpoint log file.")

        except:
            print("Couldn't properly parse rate setpoint data. Continuing on.")


def create_time_series_from_csv_logs(ahrs_type_full=True):
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
        with open(filename, 'w') as file:
            file.write(header_str)

            for key_val in data.keys():
                if __debug__:
                    ahrsData = data[key_val]['ahrsMeas']
                    motorData = data[key_val]['motorCMD']
                    angleData = data[key_val]['angleSet']
                    rateData = data[key_val]['rateSet']

                line = str(key_val) + ',' + \
                       stringify(data[key_val]['ahrsMeas']) + ',' + \
                       stringify(data[key_val]['motorCMD']) + ',' + \
                       stringify(data[key_val]['angleSet']) + ',' + \
                       stringify(data[key_val]['rateSet']) + '\n'

                file.write(line)

    print("Starting creation of time series log file...")

    motor_data              = pd.read_csv(output_motorCSVFile, header=None)
    ahrs_data               = pd.read_csv(output_ahrsCSVFile, header=None)
    angle_setpoint_data     = pd.read_csv(output_angleSetpointCSVFile, header=None)
    rate_setpoint_data      = pd.read_csv(output_rateSetpointCSVFile, header=None)

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

    # ------------------------------
    # Grab every possible recorded freeRTOS tick value and
    # initialize the time_series_data dictionary to zero
    # ------------------------------
    motor_ticks             = np.array(motor_data[time_col]).reshape(-1, 1)
    ahrs_ticks              = np.array(ahrs_data[time_col]).reshape(-1, 1)
    angle_setpoint_ticks    = np.array(angle_setpoint_data[time_col]).reshape(-1, 1)
    rate_setpoint_ticks     = np.array(rate_setpoint_data[time_col]).reshape(-1, 1)

    all_recorded_ticks = np.r_[motor_ticks, ahrs_ticks, angle_setpoint_ticks, rate_setpoint_ticks]

    print("\tCreating time series structure...")
    time_series_data = {}
    for tic in all_recorded_ticks[:, 0]:
        if tic not in time_series_data:
            time_series_data[tic] = {
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
    # """
    print("\tFilling motor data...")
    for idx in range(0, len(motor_ticks)):
        tic = motor_data[time_col][idx]
        time_series_data[tic]['motorCMD'] = np.array([motor_data[m1_col][idx],
                                                      motor_data[m2_col][idx],
                                                      motor_data[m3_col][idx],
                                                      motor_data[m4_col][idx]])

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

    # """
    print("\tFilling angle setpoint data...")
    for idx in range(0, len(angle_setpoint_ticks)):
        tic = angle_setpoint_data[time_col][idx]
        time_series_data[tic]['angleSet'] = np.array([angle_setpoint_data[pitch_col][idx],
                                                      angle_setpoint_data[roll_col][idx],
                                                      angle_setpoint_data[yaw_col][idx]])

    print("\tFilling rate setpoint data...")
    for idx in range(0, len(rate_setpoint_ticks)):
        tic = rate_setpoint_data[time_col][idx]
        time_series_data[tic]['rateSet'] = np.array([rate_setpoint_data[pitch_col][idx],
                                                     rate_setpoint_data[roll_col][idx],
                                                     rate_setpoint_data[yaw_col][idx]])

    # As a sanity check for later, make sure the data is logged before interpolation
    print("\tWriting filled data to file")
    write_time_series_to_file(filename=output_TSBlankSpaceCSVFile,
                              header_str=header_timeSeriesCSV,
                              data=time_series_data)

    # ------------------------------
    # Linear interpolate the empty spaces left by ahrsData
    # ------------------------------
    #"""
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
    # """

    # ------------------------------
    # Fill in the zero fields for the PID controller
    # ------------------------------
    print("\tInterpolating PID...")
    keys = list(time_series_data.keys())
    keys.sort(key=int)

    update_rate_ms = 8

    # Start with the angle controller for pitch, roll, yaw
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
    write_time_series_to_file(filename=output_TSInterpCSVFIle,
                              header_str=header_timeSeriesCSV,
                              data=time_series_data)

    print("Done")


def smooth_time_series_data():
    """
    Uses several matlab specific commands to smooth out the time series data. Currently
    hardcoded for simplicity
    :return:
    """
    print("Starting Matlab Engine")
    eng = matlab.engine.start_matlab()
    eng.addpath(r'C:\git\GitHub\ValkyrieRNN\Scripts\Matlab', nargout=0)

    print("Executing data smoother")
    eng.SmoothTimeSeriesLog(nargout=0)
    print("Done")


if __name__ == "__main__":
    input_ahrsLogFile_minimal   = "../DroneData/raw/ahrsLogMinimal.dat"
    input_ahrsLogFile_full      = "../DroneData/raw/ahrsLogFull.dat"
    input_motorLogFile          = "../DroneData/raw/motorLog.dat"
    input_angleSetpointLogFile  = "../DroneData/raw/angleSetpoints.dat"
    input_rateSetpointLogFile   = "../DroneData/raw/rateSetpoints.dat"

    output_ahrsCSVFile          = "../DroneData/csv/ahrsLog.csv"
    output_motorCSVFile         = "../DroneData/csv/motorLog.csv"
    output_angleSetpointCSVFile = "../DroneData/csv/angleSetpoints.csv"
    output_rateSetpointCSVFile  = "../DroneData/csv/rateSetpoints.csv"
    output_TSBlankSpaceCSVFile  = "../DroneData/csv/timeSeriesDataRaw.csv"
    output_TSInterpCSVFIle      = "../DroneData/csv/timeSeriesDataInterpolated.csv"

    matlab_smoothTimeSeriesFile = "Matlab/SmoothTimeSeriesLog.m"

    raw_data_2_csv()
    create_time_series_from_csv_logs()
    smooth_time_series_data()
