from __future__ import division, print_function, absolute_import

from src.Inference import DroneModel
from src.IPC import TCPSocket
from src.TensorFlowModels import ModelConfig

import os
import tflearn
import tensorflow
import src.TensorFlowModels as TFModels
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

if __name__ == "__main__":
    HOST = '127.0.0.1'
    PORT = 50007
    conn = TCPSocket(host_ip=HOST, port_num=PORT)

    conn.connect()

    curr_path = os.path.dirname(os.path.abspath(__file__))

    # -------------------------------
    # Load the model configuration files
    # -------------------------------
    euler_cfg = ModelConfig()
    euler_cfg.load('euler_model_cfg.csv')
    # euler_ckpt_path = 'Checkpoints/DroneRNN_Ver3/EulerModel/LastResult/' + euler_cfg.model_name + '.ckpt'
    euler_ckpt_path = 'Checkpoints/DroneRNN_Ver3/EulerModel/SmallTesting/' + euler_cfg.model_name + '.ckpt'

    gyro_cfg = ModelConfig()
    gyro_cfg.load('gyro_model_cfg.csv')
    # gyro_ckpt_path = 'Checkpoints/DroneRNN_Ver3/GyroModel/LastResult/' + gyro_cfg.model_name + '.ckpt'
    gyro_ckpt_path = 'Checkpoints/DroneRNN_Ver3/GyroModel/SmallTesting/' + gyro_cfg.model_name + '.ckpt'

    # -------------------------------
    # Setup the simulation
    # -------------------------------
    sys = DroneModel(tf_euler_chkpt_path=euler_ckpt_path, tf_gyro_chkpt_path=gyro_ckpt_path)
    sys.initialize(euler_cfg_path='euler_model_cfg.csv', gyro_cfg_path='gyro_model_cfg.csv')

    sys.set_pitch_ctrl_pid(kp_angle=2.5, ki_angle=3.0, kd_angle=0.01,
                           kp_rate=0.9, ki_rate=4.0, kd_rate=0.05)

    sys.set_roll_ctrl_pid(kp_angle=2.5, ki_angle=3.0, kd_angle=0.01,
                          kp_rate=0.9, ki_rate=4.0, kd_rate=0.05)

    sys.set_yaw_ctrl_pid(0, 0, 0, 0, 0, 0)

    pid_settings = sys.pitch_pid
    print("Simulating with pitch pid values:")
    print("Kp =", pid_settings['angles'][0])
    print("Ki =", pid_settings['angles'][1])
    print("Kd =", pid_settings['angles'][2])

    # -------------------------------
    # Execute the simulation
    # -------------------------------
    euler_data, gyro_data, setpoint_data, motor_history = \
        sys.simulate_pitch_step(step_input_delta=10.0, step_enable_t0=10, num_sim_steps=1000)

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
    plt.plot(euler_data[0, :], 'r-', label='Pitch')
    plt.plot(euler_data[1, :], 'g-', label='Roll')
    plt.plot(setpoint_data[0, :], 'c-', label='Pitch Setpoint')
    plt.legend()

    # Plot the last set of motor data sent in to the NN
    plt.figure(figsize=(16, 4))
    plt.suptitle('Motor Cmds NN Simulation Output')
    plt.plot(motor_history[0, :], 'r-', label='M1')
    plt.plot(motor_history[1, :], 'g-', label='M2')
    plt.plot(motor_history[2, :], 'b-', label='M3')
    plt.plot(motor_history[3, :], 'o-', label='M4')
    #plt.plot(angle_data[0, :], 'c-', label='Pitch Setpoint')
    plt.legend()

    # plt.show()

    # ----------------------------
    # Give the data to Matlab for post-processing
    # ----------------------------

    # ----------------------------
    # Return results back to the GA software
    # ----------------------------
    # For now just report back dummy data






