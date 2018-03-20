# This file contains descriptions for the various kinds of data structures utilized
# in the ValkyrieFCS software to log drone flight data. The classes are only intended
# to be utilized in conjunction with the RawData2CSV file for parsing the raw binary
# data dumps on the sd card into a more readable csv file.

import struct


class SDLogAHRSFull:
    def __init__(self):
        self.structSizeInBytes = 52
        self.dataFormatString = 'Iffffffffffff'  # Should be 12 f's

        # Number of freeRTOS ticks elapsed @ measurement
        self.tickTime = 0

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
        self.rawData = ()

    def unpack_raw_hex(self, hex_data):
        assert(len(hex_data) == self.structSizeInBytes)
        self.rawData = struct.unpack(self.dataFormatString, hex_data)

        self.euler_deg_pitch = self.rawData[0]
        self.euler_deg_roll = self.rawData[1]
        self.euler_deg_yaw = self.rawData[2]
        self.accel_x = self.rawData[3]
        self.accel_y = self.rawData[4]
        self.accel_z = self.rawData[5]
        self.gyro_x = self.rawData[6]
        self.gyro_y = self.rawData[7]
        self.gyro_z = self.rawData[8]
        self.mag_x = self.rawData[9]
        self.mag_y = self.rawData[10]
        self.mag_z = self.rawData[11]

    def as_csv(self):
        csv_stringified_data = [(str(x) + ',') for x in self.rawData]
        csv_stringified_data[-1] = csv_stringified_data[-1].replace(",", "\n")

        return ''.join(csv_stringified_data)


class SDLogAHRSMinimal:
    def __init__(self):
        self.structSizeInBytes = 16
        self.dataFormatString = 'Ifff'
        self.rawData = ()

        # Number of freeRTOS ticks elapsed @ measurement
        self.tickTime = 0

        # Degrees
        self.euler_deg_pitch = 0.0
        self.euler_deg_roll = 0.0
        self.euler_deg_yaw = 0.0

    def unpack_raw_hex(self, hex_data):
        assert(len(hex_data) == self.structSizeInBytes)

        self.rawData = struct.unpack(self.dataFormatString, hex_data)

        self.tickTime = self.rawData[0]
        self.euler_deg_pitch = self.rawData[1]
        self.euler_deg_roll = self.rawData[2]
        self.euler_deg_yaw = self.rawData[3]

    def as_csv(self):
        csv_stringified_data = [(str(x) + ',') for x in self.rawData]
        csv_stringified_data[-1] = csv_stringified_data[-1].replace(",", "\n")

        return ''.join(csv_stringified_data)


class SDLogMotor:
    def __init__(self):
        self.structSizeInBytes = 12
        self.dataFormatString = 'IHHHH'
        self.rawData = ()

        # Number of freeRTOS ticks elapsed @ measurement
        self.tickTime = 0

        # Raw motor command signal time high in uS
        self.m1 = 0
        self.m2 = 0
        self.m3 = 0
        self.m4 = 0

    def unpack_raw_hex(self, hex_data):
        assert (len(hex_data) == self.structSizeInBytes)

        self.rawData = struct.unpack(self.dataFormatString, hex_data)

        self.tickTime = self.rawData[0]
        self.m1 = self.rawData[1]
        self.m2 = self.rawData[2]
        self.m3 = self.rawData[3]
        self.m4 = self.rawData[4]

    def as_csv(self):
        csv_stringified_data = [(str(x) + ',') for x in self.rawData]
        csv_stringified_data[-1] = csv_stringified_data[-1].replace(",", "\n")

        return ''.join(csv_stringified_data)


class SDLogAngleSetpoint:
    def __init__(self):
        self.structSizeInBytes = 16
        self.dataFormatString = 'Ifff'
        self.rawData = ()

        # Number of freeRTOS ticks elapsed @ measurement
        self.tickTime = 0

        # Raw motor command signal time high in uS
        self.pitch_angle_setpoint = 0
        self.roll_angle_setpoint = 0
        self.yaw_angle_setpoint = 0

    def unpack_raw_hex(self, hex_data):
        assert (len(hex_data) == self.structSizeInBytes)

        self.rawData = struct.unpack(self.dataFormatString, hex_data)

        self.tickTime = self.rawData[0]
        self.pitch_angle_setpoint = self.rawData[1]
        self.roll_angle_setpoint = self.rawData[2]
        self.yaw_angle_setpoint = self.rawData[3]

    def as_csv(self):
        csv_stringified_data = [(str(x) + ',') for x in self.rawData]
        csv_stringified_data[-1] = csv_stringified_data[-1].replace(",", "\n")

        return ''.join(csv_stringified_data)


class SDLogRateSetpoint:
    def __init__(self):
        self.structSizeInBytes = 16
        self.dataFormatString = 'Ifff'
        self.rawData = ()

        # Number of freeRTOS ticks elapsed @ measurement
        self.tickTime = 0

        # Raw motor command signal time high in uS
        self.pitch_rate_setpoint = 0
        self.roll_rate_setpoint = 0
        self.yaw_rate_setpoint = 0

    def unpack_raw_hex(self, hex_data):
        assert (len(hex_data) == self.structSizeInBytes)

        self.rawData = struct.unpack(self.dataFormatString, hex_data)

        self.tickTime = self.rawData[0]
        self.pitch_rate_setpoint = self.rawData[1]
        self.roll_rate_setpoint = self.rawData[2]
        self.yaw_rate_setpoint = self.rawData[3]

    def as_csv(self):
        csv_stringified_data = [(str(x) + ',') for x in self.rawData]
        csv_stringified_data[-1] = csv_stringified_data[-1].replace(",", "\n")

        return ''.join(csv_stringified_data)
