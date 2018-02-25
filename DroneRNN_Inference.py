from __future__ import division, print_function, absolute_import

from src.Inference import DroneModel
from src.IPC import TCPSocket

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
    nn_path = curr_path + '\Checkpoints\DroneRNN_Ver2\BestResult\\'
    nn_model = 'model.tfl.ckpt8995'

    sys = DroneModel(tf_cell_type='rnn', tf_model_name=nn_model, tf_chkpt_path=nn_path)
    sys.initialize()
