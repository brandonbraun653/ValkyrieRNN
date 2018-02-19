# This file takes the output of RawData2CSV and filters through it to produce a single csv file with
# concatenated sensor data at each time step. Because the freeRTOS implementation has slight delays
# between tasks, there are "gaps" in the information that must be filled in.

import os
import pandas as pd
import numpy as np

# Setup the motor data
motorLogFilename = 'C:/git/GitHub/ValkyrieRNN/DroneData/motorLog.csv'
timeCol = 0
m1Col = 1
m2Col = 2
m3Col = 3
m4Col = 4

motorData = pd.read_csv(motorLogFilename, header=None)


# Set up the AHRS data
ahrsLogFilename = '../DroneData/ahrsLogMinimal.csv'
pitchCol = 1
rollCol = 2
yawCol = 3

ahrsData = pd.read_csv(ahrsLogFilename, header=None)


# Contains all info at each time step from given input files
timeSeriesData = {}
timeSeriesDataFilename = '../DroneData/timeSeriesData.csv'

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
keys = timeSeriesData.keys()

for i in range(0, len(timeSeriesData.keys())):
    tic = keys[i]

    ahrs = timeSeriesData[tic]['ahrsMeas']
    ahrsSum = ahrs[0] + ahrs[1] + ahrs[2]

    # Predict the missing value by using an average
    if ahrsSum == 0:
        ahrsPrev = timeSeriesData[keys[i-1]]['ahrsMeas']

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
            new_value = np.array(timeSeriesData[keys[j-1]]['ahrsMeas']) + ahrsDeltaDiff

            timeSeriesData[keys[j]]['ahrsMeas'][0] = new_value[0]
            timeSeriesData[keys[j]]['ahrsMeas'][1] = new_value[1]
            timeSeriesData[keys[j]]['ahrsMeas'][2] = new_value[2]

# Fill in the zero fields for motor commands (values don't change in MCU until update)
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
