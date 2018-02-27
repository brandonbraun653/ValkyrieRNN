from __future__ import division, print_function, absolute_import

from src.Inference import DroneModel
from src.IPC import TCPSocket
from src.TensorFlowModels import ModelConfig

import os
import tflearn
import tensorflow
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as graph



if __name__ == "__main__":
    HOST = '127.0.0.1'
    PORT = 50007
    conn = TCPSocket(host_ip=HOST, port_num=PORT)

    curr_path = os.path.dirname(os.path.abspath(__file__))
    nn_path = curr_path + '\Checkpoints\DroneRNN_Ver3\BestResult\\'
    nn_model = 'model.tfl.ckpt8995'

    # -------------------------------
    # Load the model configuration files
    # -------------------------------
    euler_cfg = ModelConfig()
    euler_cfg.load('euler_model_cfg.csv')

    gyro_cfg = ModelConfig()
    gyro_cfg.load('gyro_model_cfg.csv')

    # -------------------------------
    # Create the NN models
    # -------------------------------


    sys = DroneModel(tf_cell_type='rnn', tf_model_name=nn_model, tf_chkpt_path=nn_path)
    sys.initialize()
