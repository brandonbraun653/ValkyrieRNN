from __future__ import division, print_function, absolute_import

from src.Inference import DroneModel
from src.InterProcessComm import TCPSocket


import os
import tflearn
import tensorflow
import src.TensorFlowModels as TFModels
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import mmap
import struct

if __name__ == "__main__":
    HOST = '127.0.0.1'
    PORT = 50007
    conn = TCPSocket(host_ip=HOST, port_num=PORT)

    # -------------------------------
    # Setup the simulation
    # -------------------------------
    sys = DroneModel(euler_cfg_path='euler_model_cfg.csv', gyro_cfg_path='gyro_model_cfg.csv')
    sys.initialize(ahrs_sample_freq=500, pid_update_freq=125)

    sim_start_time = 0.0
    sim_end_time = 10.0
    step_mag = 10.0
    step_enable = 0.5

    kp_ang = 6.5
    ki_ang = 8.0
    kd_ang = 0.1


    sys.set_pitch_ctrl_pid(kp_angle=kp_ang, ki_angle=ki_ang, kd_angle=kd_ang,
                           kp_rate=0.9, ki_rate=4.0, kd_rate=0.05)

    sys.set_roll_ctrl_pid(kp_angle=2.5, ki_angle=3.0, kd_angle=0.01,
                          kp_rate=0.9, ki_rate=4.0, kd_rate=0.05)

    sys.set_yaw_ctrl_pid(0, 0, 0, 0, 0, 0)

    pid_settings = sys.pitch_pid
    print("Simulating with pitch pid values:")
    print("Kp =", pid_settings['angles'][0])
    print("Ki =", pid_settings['angles'][1])
    print("Kd =", pid_settings['angles'][2])
    print("Start Time: ", sim_start_time)
    print("End Time: ", sim_end_time)

    # -------------------------------
    # Execute the simulation
    # -------------------------------
    # TODO: I think I may need to reset the model dynamics between each simulation.
    # TODO: Print out the simulation parameters too, like num steps, and maybe a few performance specs

    results = sys.simulate_pitch_step(start_time=sim_start_time,
                                      end_time=sim_end_time,
                                      step_input_delta=step_mag,
                                      step_ON_pct=step_enable)

    time = results['time']
    euler_data = results['euler_output']
    gyro_data = results['gyro_output']
    setpoint_data = results['angle_setpoints']

    euler_model_input = results['euler_input']
    pitch_ctrl = results['pitch_ctrl_output']
    roll_ctrl = results['roll_ctrl_output']

    # Plot Gyro Data
    plt.figure(figsize=(16, 4))
    plt.suptitle('Gyro NN Simulation Output')
    plt.plot(gyro_data[0, :], 'r-', label='GX')
    plt.plot(gyro_data[1, :], 'g-', label='GY')
    plt.plot(setpoint_data[0, :], 'c-', label='Pitch Setpoint')
    plt.legend()

    # Plot Euler Data
    plt.figure(figsize=(16, 4))
    plt.suptitle('Euler NN Simulation Output')
    plt.plot(time, euler_data[0, :], 'r-', label='Pitch')
    plt.plot(time, euler_data[1, :], 'g-', label='Roll')
    plt.plot(time, setpoint_data[0, :], 'c-', label='Pitch Setpoint')
    plt.legend()

    # Plot the last set of motor data sent in to the NN
    plt.figure(figsize=(16, 4))
    plt.suptitle('Euler Model Input Signal')
    plt.plot(euler_model_input[0, :], 'r-', label='M1')
    plt.plot(euler_model_input[1, :], 'g-', label='M2')
    plt.plot(euler_model_input[2, :], 'b-', label='M3')
    plt.plot(euler_model_input[3, :], 'c-', label='M4')
    plt.legend()

    # Plot the input to the model
    plt.figure(figsize=(16, 4))
    plt.suptitle('Pitch/Roll Controller Signal')
    plt.plot(pitch_ctrl, 'r-', label='Pitch')
    plt.plot(roll_ctrl, 'b-', label='Roll')
    plt.legend()

    plt.show()







