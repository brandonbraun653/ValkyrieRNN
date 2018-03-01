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

    curr_path = os.path.dirname(os.path.abspath(__file__))

    # -------------------------------
    # Load the model configuration files
    # -------------------------------
    euler_cfg = ModelConfig()
    euler_cfg.load('euler_model_cfg.csv')
    euler_ckpt_path = 'Checkpoints/DroneRNN_Ver3/EulerModel/LastResult/' + euler_cfg.model_name + '.ckpt'

    gyro_cfg = ModelConfig()
    gyro_cfg.load('gyro_model_cfg.csv')
    gyro_ckpt_path = 'Checkpoints/DroneRNN_Ver3/GyroModel/LastResult/' + gyro_cfg.model_name + '.ckpt'

    # -------------------------------
    # Create the NN models
    # -------------------------------
    sys = DroneModel(tf_euler_chkpt_path=euler_ckpt_path, tf_gyro_chkpt_path=gyro_ckpt_path)
    sys.initialize(euler_cfg_path='euler_model_cfg.csv', gyro_cfg_path='gyro_model_cfg.csv')

    sys.set_pitch_ctrl_pid(kp_angle=2.5, ki_angle=3.0, kd_angle=0.01,
                           kp_rate=0.9, ki_rate=4.0, kd_rate=0.05)

    sys.set_roll_ctrl_pid(kp_angle=2.5, ki_angle=3.0, kd_angle=0.01,
                          kp_rate=0.9, ki_rate=4.0, kd_rate=0.05)

    sys.set_yaw_ctrl_pid(0, 0, 0, 0, 0, 0)

    output = sys.simulate_pitch_step(step_size=10.0, sim_length= 250)








