import struct
import os

class AHRSData():
    def __init__(self):
        # These variables must reflect the latest structure definition of SD_LOG_AHRS_t
        # in the ValkyrieFCS datatypes.hpp file (line 106)
        self.numParams = 12
        self.floatByteSize = 4
        self.structSizeInBytes = (self.numParams * self.floatByteSize)
        self.dataFormatString = 'f' * self.numParams

        # Degrees
        self.euler_deg_pitch = 0.0
        self.euler_deg_roll = 0.0
        self.euler_deg_yaw = 0.0

        # M/S^2
        self.accel_x = 0.0
        self.accel_y = 0.0
        self.accel_z = 0.0

        # DPS
        self.gyro_x = 0.0
        self.gyro_y = 0.0
        self.gyro_z = 0.0

        # GAUSS
        self.mag_x = 0.0
        self.mag_y = 0.0
        self.mag_z = 0.0

        # Raw List Format
        self.rawFloats = ()

    def unpack_raw_hex(self, hex_data):
        """
        Converts raw hex data into the euler angle, accelerometer, gyroscope, and
        magnetometer measurements
        """

        assert(len(hex_data) == self.structSizeInBytes)
        self.rawFloats = struct.unpack(self.dataFormatString, hex_data)

        self.euler_deg_pitch = self.rawFloats[0]
        self.euler_deg_roll = self.rawFloats[1]
        self.euler_deg_yaw = self.rawFloats[2]
        self.accel_x = self.rawFloats[3]
        self.accel_y = self.rawFloats[4]
        self.accel_z = self.rawFloats[5]
        self.gyro_x = self.rawFloats[6]
        self.gyro_y = self.rawFloats[7]
        self.gyro_z = self.rawFloats[8]
        self.mag_x = self.rawFloats[9]
        self.mag_y = self.rawFloats[10]
        self.mag_z = self.rawFloats[11]

    def as_csv(self):
        csv_stringified_data = [(str(x) + ',') for x in self.rawFloats]
        csv_stringified_data[-1] = csv_stringified_data[-1].replace(",", "\n")

        return ''.join(csv_stringified_data)


rawAHRS = []

flight_log_file = "C:/git/GitHub/ValkyrieRNN/DroneData/flightLog.dat"
flight_log_size = os.path.getsize(flight_log_file)


with open(flight_log_file, "rb") as file:
    bytesRead = 0

    while bytesRead < flight_log_size:
        measurement = AHRSData()
        measurement.unpack_raw_hex(file.read(measurement.structSizeInBytes))

        rawAHRS.append(measurement)

        bytesRead += measurement.structSizeInBytes


print("Done reading file.")


flight_log_csv = "C:/git/GitHub/ValkyrieRNN/DroneData/flightLog.csv"

with open(flight_log_csv, 'w') as file:
    [file.write(rawAHRS[x].as_csv()) for x in range(len(rawAHRS))]

print("Done writing to CSV.")


