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
import matplotlib.pyplot as graph

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



    sys.TESTFUNC()






    # ---------------------------------------
    # Plot some data for the user to see how well training went
    # ---------------------------------------
    # X = []
    # Y = []
    #
    # for i in range(len(input[:, 0]) - trainLen, len(input[:, 0])-cfg.input_depth):
    #     X.append(input[i:i + cfg.input_depth, 0:cfg.input_size])
    #     Y.append(output[i + cfg.input_depth, 0:cfg.output_size])
    #
    # X = np.reshape(X, [-1, cfg.input_depth, cfg.input_size])
    # Y = np.reshape(Y, [-1, cfg.output_size])
    #
    # predictY = model.predict(X)

    # Plot the results
    # print("Plotting Sample Outputs")
    # imgSavePath = "Checkpoints/DroneRNN_Ver3/EulerModel/Images/"

    # PITCH
    # plt.figure(figsize=(16, 4))
    # plt.suptitle('Pitch Actual Vs Predicted')
    # plt.plot(Y[:, 0], 'r-', label='Actual')
    # plt.plot(predictY[:, 0], 'g-', label='Predicted')
    # plt.legend()
    # plt.savefig(imgSavePath+'pitch.png')

    # ROLL
    # plt.figure(figsize=(16, 4))
    # plt.suptitle('Roll Actual vs Predicted')
    # plt.plot(Y[1, :], 'r-', label='Actual')
    # plt.plot(predictY[1, :], 'g-', label='Predicted')
    # plt.legend()
    # plt.savefig(imgSavePath + 'roll.png')

    # ---------------------------------------
    # Plot some data for the user to see how well training went
    # ---------------------------------------
    # X = []
    # Y = []
    #
    # for i in range(len(input[:, 0]) - trainLen, len(input[:, 0]) - cfg.input_depth):
    #     X.append(input[i:i + cfg.input_depth, 0:cfg.input_size])
    #     Y.append(output[i + cfg.input_depth, 0:cfg.output_size])
    #
    # X = np.reshape(X, [-1, cfg.input_depth, cfg.input_size])
    # Y = np.reshape(Y, [-1, cfg.output_size])
    #
    # predictY = model.predict(X)
    #
    # # Plot the results
    # print("Plotting Sample Outputs")
    # imgSavePath = "Checkpoints/DroneRNN_Ver3/GyroModel/Images/"
    #
    # # GYRO X
    # plt.figure(figsize=(16, 4))
    # plt.suptitle('GX Actual vs Predicted')
    # plt.plot(Y[2, :], 'r-', label='Actual')
    # plt.plot(predictY[2, :], 'g-', label='Predicted')
    # plt.legend()
    # plt.savefig(imgSavePath + 'gx.png')
    #
    # # GYRO Y
    # plt.figure(figsize=(16, 4))
    # plt.suptitle('GY Actual vs Predicted')
    # plt.plot(Y[3, :], 'r-', label='Actual')
    # plt.plot(predictY[3, :], 'g-', label='Predicted')
    # plt.legend()
    # plt.savefig(imgSavePath + 'gy.png')
    #
    # # GYRO Z
    # plt.figure(figsize=(16, 4))
    # plt.suptitle('GZ Actual vs Predicted')
    # plt.plot(Y[4, :], 'r-', label='Actual')
    # plt.plot(predictY[4, :], 'g-', label='Predicted')
    # plt.legend()
    # plt.savefig(imgSavePath + 'gz.png')