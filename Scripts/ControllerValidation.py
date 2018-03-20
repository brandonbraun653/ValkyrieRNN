# This file is just a quick script to validate that the virtual PID controller used with the
# neural network model actually generates the same response as the real PID controller flying
# the quadrotor.

import numpy as np
import pandas as pd

import matplotlib; matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import src.MotorController as MotorController

from src.Miscellaneous import bcolors
from src.ControlSystem import AxisController
from scipy.interpolate import interp1d

# Location for the flight log file
log_data_path = 'G:/Projects/ValkyrieRNN/Data/ControlSystemTests/csv/Test1/'
log_file = 'timeSeriesDataSmoothed.csv'

# PID Settings used in ^^^
ANGLE_KP_PITCH = 1.5
ANGLE_KI_PITCH = 3.0
ANGLE_KD_PITCH = 0.01
RATE_KP_PITCH = 0.9
RATE_KI_PITCH = 4.0
RATE_KD_PITCH = 0.05

ANGLE_KP_ROLL = 1.5
ANGLE_KI_ROLL = 3.0
ANGLE_KD_ROLL = 0.01
RATE_KP_ROLL = 0.9
RATE_KI_ROLL = 4.0
RATE_KD_ROLL = 0.05

# PID Output Ranges Used
RATE_OUTPUT_RANGE = 100.0
MOTOR_OUTPUT_RANGE = 500.0

# Sampling Rates Used
AHRS_UPDATE_FREQ = 500
PID_UPDATE_FREQ = 125

# Choose what plots to show
PLOT_INPUT_DATA = False

# Controller Directions
REVERSE = True
DIRECT = False

# Controller setup according to parameters above
pitch_ctrl = AxisController(angular_rate_range=RATE_OUTPUT_RANGE,
                            motor_cmd_range=MOTOR_OUTPUT_RANGE,
                            angle_direction=REVERSE,
                            rate_direction=DIRECT,
                            sample_time_ms=int(1000.0/PID_UPDATE_FREQ))
pitch_ctrl.update_angle_pid(kp=ANGLE_KP_PITCH, ki=ANGLE_KI_PITCH, kd=ANGLE_KD_PITCH)
pitch_ctrl.update_rate_pid(kp=RATE_KP_PITCH, ki=RATE_KI_PITCH, kd=RATE_KD_PITCH)


roll_ctrl = AxisController(angular_rate_range=RATE_OUTPUT_RANGE,
                           motor_cmd_range=MOTOR_OUTPUT_RANGE,
                           angle_direction=DIRECT,
                           rate_direction=DIRECT,
                           sample_time_ms=int(1000.0/PID_UPDATE_FREQ))

roll_ctrl.update_angle_pid(kp=ANGLE_KP_ROLL, ki=ANGLE_KI_ROLL, kd=ANGLE_KD_ROLL)
roll_ctrl.update_rate_pid(kp=RATE_KP_ROLL, ki=RATE_KI_ROLL, kd=RATE_KD_ROLL)


# -------------------
# Pull in all reference data
# -------------------
raw_data = pd.read_csv(log_data_path+log_file)
tick_tock = np.array(raw_data['rtosTick']).reshape(-1, 1)

gx = np.array(raw_data['gx']).reshape(-1, 1)
gy = np.array(raw_data['gy']).reshape(-1, 1)
gz = np.array(raw_data['gz']).reshape(-1, 1)

pitch = np.array(raw_data['pitch']).reshape(-1, 1)
roll = np.array(raw_data['roll']).reshape(-1, 1)

m1CMD = np.array(raw_data['m1CMD']).reshape(-1, 1)
m2CMD = np.array(raw_data['m2CMD']).reshape(-1, 1)
m3CMD = np.array(raw_data['m3CMD']).reshape(-1, 1)
m4CMD = np.array(raw_data['m4CMD']).reshape(-1, 1)

pitch_angle_setpoint = np.array(raw_data['asp']).reshape(-1, 1)
pitch_rate_setpoint = np.array(raw_data['rsp']).reshape(-1, 1)

roll_angle_setpoint = np.array(raw_data['asr']).reshape(-1, 1)
roll_rate_setpoint = np.array(raw_data['rsr']).reshape(-1, 1)


# -------------------
# Plot a few values
# -------------------
if PLOT_INPUT_DATA:
    lw = 0.8
    plt.figure(figsize=(32, 18))
    plt.plot(tick_tock, pitch_angle_setpoint, 'g-', label='Pitch Setpoint', linewidth=lw)
    plt.plot(tick_tock, pitch_rate_setpoint, 'c-', label='Pitch Rate Setpoint', linewidth=lw)
    plt.plot(tick_tock, pitch, 'b-', label='Pitch Angle', linewidth=lw)
    plt.plot(tick_tock, gy, 'r--', label='Pitch Rate', linewidth=lw)
    plt.legend()

    plt.figure(figsize=(32, 18))
    plt.plot(tick_tock, roll_angle_setpoint, 'g-', label='Roll Setpoint', linewidth=lw)
    plt.plot(tick_tock, roll_rate_setpoint, 'c-', label='Roll Rate Setpoint', linewidth=lw)
    plt.plot(tick_tock, roll, 'b-', label='Roll Angle', linewidth=lw)
    plt.plot(tick_tock, -gx, 'r--', label='Roll Rate', linewidth=lw)
    plt.legend()
    plt.show()

# -------------------
# Simulate the virtual PID controller
# -------------------
sim_len = range(1, np.size(tick_tock, axis=0))

motor_log = np.zeros([4, len(sim_len)+1])
v_pitch_cmd_log = np.zeros([1, len(sim_len) + 1])
v_pitch_rate_log = np.zeros([1, len(sim_len)+1])
v_roll_cmd_log = np.zeros([1, len(sim_len) + 1])
v_roll_rate_log = np.zeros([1, len(sim_len)+1])

last_time = tick_tock[0]
pid_update_ms = (1000.0/PID_UPDATE_FREQ)

pitch_cmd = 0
roll_cmd = 0

for step in sim_len:

    current_time = tick_tock[step]
    dt = current_time - last_time

    if dt % pid_update_ms == 0:
        last_time = current_time

        # -------------------
        # Update pitch controller
        # -------------------
        pitch_ctrl.angle_setpoint = pitch_angle_setpoint[step-1, 0]
        pitch_ctrl.angle_feedback = pitch[step, 0]
        pitch_ctrl.rate_feedback = gy[step, 0]

        pitch_ctrl.compute(use_rate_control=True)

        v_pitch_cmd_log[:, step] = pitch_ctrl.controller_output
        v_pitch_rate_log[:, step] = pitch_ctrl.angular_rate_desired

        # -------------------
        # Update roll controller
        # -------------------
        roll_ctrl.angle_setpoint = roll_angle_setpoint[step-1, 0]
        roll_ctrl.angle_feedback = roll[step, 0]
        roll_ctrl.rate_feedback = -gx[step, 0]

        roll_ctrl.compute(use_rate_control=True)

        v_roll_cmd_log[:, step] = roll_ctrl.controller_output
        v_roll_rate_log[:, step] = roll_ctrl.angular_rate_desired

        pitch_cmd = pitch_ctrl.controller_output
        roll_cmd = roll_ctrl.controller_output

    # -------------------
    # Update roll controller
    # -------------------
    motor_signal = MotorController.generate_motor_signals(1160, pitch_cmd, roll_cmd, 0)
    motor_log[:, step] = motor_signal[:, 0]


# -------------------
# Plot the output comparison
# -------------------
lw = 0.8

# Compare PITCH ANGULAR_RATE_DESIRED output
plt.figure(figsize=(32, 18))
plt.plot(tick_tock, v_pitch_rate_log[0, :], 'g-', label='Virtual Pitch Rate Setpoint', linewidth=lw)
plt.plot(tick_tock, pitch_rate_setpoint, 'c-', label='Actual Pitch Rate Setpoint', linewidth=lw)
plt.legend()

# Compare m1CMD output
plt.figure(figsize=(32, 18))
plt.plot(tick_tock, motor_log[0, :], 'g-', label='Virtual M1CMD', linewidth=lw)
plt.plot(tick_tock, m1CMD, 'c-', label='Actual M1CMD', linewidth=lw)
plt.legend()

plt.figure(figsize=(32, 18))
plt.plot(tick_tock, v_pitch_cmd_log[0, :], 'g-', label='Virtual Pitch CMD', linewidth=lw)
plt.plot(tick_tock, v_roll_cmd_log[0, :], 'b-', label='Virtual Roll CMD', linewidth=lw)
plt.legend()

plt.show()

