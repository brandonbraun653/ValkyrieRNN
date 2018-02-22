from Scripts import DataStructures

import os
import pandas as pd
import numpy as np
import matlab.engine

input_ahrsLogFile = "DroneData/ahrsLogMinimal.dat"
output_ahrsCSVFile = "DroneData/ahrsLogMinimal.csv"

input_motorLogFile = "DroneData/motorLog.dat"
output_motorCSVFile = "DroneData/motorLog.csv"

output_timeSeriesCSVFile = "DroneData/timeSeriesDataRaw.csv"

smoother_file = "SmoothTimeSeriesLog.m"


def raw_data_2_csv():
    """
    Takes the raw byte code from the SD Card logs and converts them into easy to
    interpret CSV files
    """
    rawAHRSData = []
    input_ahrsFileSize = os.path.getsize(input_ahrsLogFile)

    rawMotorData = []
    input_motorFileSize = os.path.getsize(input_motorLogFile)

    # Parse the AHRS data
    with open(input_ahrsLogFile, "rb") as input_file:
        bytesRead = 0

        while bytesRead < input_ahrsFileSize:
            measurement = DataStructures.SDLogAHRSMinimal()
            measurement.unpack_raw_hex(input_file.read(measurement.structSizeInBytes))

            rawAHRSData.append(measurement)

            bytesRead += measurement.structSizeInBytes

        with open(output_ahrsCSVFile, 'w') as output_file:
            [output_file.write(rawAHRSData[x].as_csv()) for x in range(len(rawAHRSData))]

    print("Done parsing AHRS log file.")

    # Parse the Motor data
    with open(input_motorLogFile, "rb") as input_file:
        bytesRead = 0

        while bytesRead < input_motorFileSize:
            measurement = DataStructures.SDLogMotor()
            measurement.unpack_raw_hex(input_file.read(measurement.structSizeInBytes))

            rawMotorData.append(measurement)

            bytesRead += measurement.structSizeInBytes

        with open(output_motorCSVFile, 'w') as output_file:
            [output_file.write(rawMotorData[x].as_csv()) for x in range(len(rawMotorData))]

    print("Done parsing Motor log file.")


def create_time_series_from_csv_logs():
    """
    This function takes the output of raw_data_2_csv and filters through it to produce a single csv file with
    concatenated sensor data at each time step. Because the freeRTOS implementation in ValkyrieFCS has slight delays
    between tasks, there are "gaps" in the information that must be filled in.
    """

    print("Starting creation of time series log file...")

    # Setup the motor data
    motorLogFilename = output_motorCSVFile
    timeCol = 0
    m1Col = 1
    m2Col = 2
    m3Col = 3
    m4Col = 4

    motorData = pd.read_csv(motorLogFilename, header=None)

    # Set up the AHRS data
    ahrsLogFilename = output_ahrsCSVFile
    pitchCol = 1
    rollCol = 2
    yawCol = 3

    ahrsData = pd.read_csv(ahrsLogFilename, header=None)

    # Contains all info at each time step from given input files
    timeSeriesData = {}
    timeSeriesDataFilename = output_timeSeriesCSVFile

    # Populate motor data
    for idx in range(0, len(motorData[timeCol])):
        tic = motorData[timeCol][idx]
        timeSeriesData[tic] = {'motorCMD': [motorData[m1Col][idx],
                                            motorData[m2Col][idx],
                                            motorData[m3Col][idx],
                                            motorData[m4Col][idx]],
                               'ahrsMeas': [0.0, 0.0, 0.0]}

    # Populate ahrs data
    for idx in range(0, len(ahrsData[timeCol])):
        tic = ahrsData[timeCol][idx]

        if tic in timeSeriesData:
            timeSeriesData[tic]['ahrsMeas'] = [ahrsData[pitchCol][idx],
                                               ahrsData[rollCol][idx],
                                               ahrsData[yawCol][idx]]
        else:
            timeSeriesData[tic] = {'motorCMD': [0, 0, 0, 0],
                                   'ahrsMeas': [ahrsData[pitchCol][idx],
                                                ahrsData[rollCol][idx],
                                                ahrsData[yawCol][idx]]}

    # Interpolate the empty spaces left by ahrsData
    keys = list(timeSeriesData.keys())
    for i in range(0, len(keys)):
        tic = keys[i]

        ahrs = timeSeriesData[tic]['ahrsMeas']
        ahrsSum = ahrs[0] + ahrs[1] + ahrs[2]

        # Predict the missing value by using an average
        if ahrsSum == 0:
            ahrsPrev = timeSeriesData[keys[i - 1]]['ahrsMeas']

            # Find the next non zero tic
            currentIdx = i
            numTics = 1
            while timeSeriesData[keys[currentIdx]]['ahrsMeas'][0] == 0.0:
                numTics += 1
                currentIdx += 1

            ahrsNext = timeSeriesData[keys[currentIdx]]['ahrsMeas']

            # Generate the delta for each zero'd field
            ahrsDeltaDiff = (np.array(ahrsNext) - np.array(ahrsPrev)) / numTics

            # Write the empty fields
            for j in range(i, currentIdx):
                new_value = np.array(timeSeriesData[keys[j - 1]]['ahrsMeas']) + ahrsDeltaDiff

                timeSeriesData[keys[j]]['ahrsMeas'][0] = new_value[0]
                timeSeriesData[keys[j]]['ahrsMeas'][1] = new_value[1]
                timeSeriesData[keys[j]]['ahrsMeas'][2] = new_value[2]

    # Fill in the zero fields for motor commands (values don't change in MCU until rtos tick update)
    motorsON = False
    lastValidCMD = []
    keysToRemove = []
    for i in range(0, len(timeSeriesData.keys())):

        currentMotorCMD = timeSeriesData[keys[i]]['motorCMD']

        if not motorsON:
            # We don't want any data from before the motors turn on, so delete
            # those lines from the final timeSeriesData
            if currentMotorCMD[0] == 0:
                keysToRemove.append(keys[i])
            else:
                motorsON = True
                lastValidCMD = currentMotorCMD
                continue

        if currentMotorCMD[0] == 0:
            timeSeriesData[keys[i]]['motorCMD'] = lastValidCMD
        else:
            lastValidCMD = currentMotorCMD

    for key in keysToRemove:
        timeSeriesData.pop(key)

    # Write all that data to a csv file
    with open(timeSeriesDataFilename, 'w') as output_file:
        header = "rtosTick,pitch,roll,yaw,m1CMD,m2CMD,m3CMD,m4CMD\n"
        output_file.write(header)

        for tic in timeSeriesData.keys():
            out_string = str(tic) + ',' + \
                         str(timeSeriesData[tic]['ahrsMeas'][0]) + ',' + \
                         str(timeSeriesData[tic]['ahrsMeas'][1]) + ',' + \
                         str(timeSeriesData[tic]['ahrsMeas'][2]) + ',' + \
                         str(timeSeriesData[tic]['motorCMD'][0]) + ',' + \
                         str(timeSeriesData[tic]['motorCMD'][1]) + ',' + \
                         str(timeSeriesData[tic]['motorCMD'][2]) + ',' + \
                         str(timeSeriesData[tic]['motorCMD'][3]) + '\n'

            output_file.write(out_string)

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
    input_ahrsLogFile = "../DroneData/ahrsLogMinimal.dat"
    output_ahrsCSVFile = "../DroneData/ahrsLogMinimal.csv"

    input_motorLogFile = "../DroneData/motorLog.dat"
    output_motorCSVFile = "../DroneData/motorLog.csv"

    output_timeSeriesCSVFile = "../DroneData/timeSeriesDataRaw.csv"

    smoother_file = "Matlab/SmoothTimeSeriesLog.m"

    raw_data_2_csv()
    create_time_series_from_csv_logs()
    smooth_time_series_data()