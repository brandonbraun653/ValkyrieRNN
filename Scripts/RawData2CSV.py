import os
from DataStructures import SDLogAHRSFull, SDLogAHRSMinimal, SDLogMotor


rawAHRSData = []
input_ahrsLogFile = "C:/git/GitHub/ValkyrieRNN/DroneData/ahrsLogMinimal.dat"
input_ahrsFileSize = os.path.getsize(input_ahrsLogFile)
output_ahrsCSVFile = "C:/git/GitHub/ValkyrieRNN/DroneData/ahrsLogMinimal.csv"

rawMotorData = []
input_motorLogFile = "C:/git/GitHub/ValkyrieRNN/DroneData/motorLog.dat"
input_motorFileSize = os.path.getsize(input_motorLogFile)
output_motorCSVFile = "C:/git/GitHub/ValkyrieRNN/DroneData/motorLog.csv"


# Parse the AHRS data
with open(input_ahrsLogFile, "rb") as input_file:
    bytesRead = 0

    while bytesRead < input_ahrsFileSize:
        measurement = SDLogAHRSMinimal()
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
        measurement = SDLogMotor()
        measurement.unpack_raw_hex(input_file.read(measurement.structSizeInBytes))

        rawMotorData.append(measurement)

        bytesRead += measurement.structSizeInBytes

    with open(output_motorCSVFile, 'w') as output_file:
        [output_file.write(rawMotorData[x].as_csv()) for x in range(len(rawMotorData))]

print("Done parsing Motor log file.")
