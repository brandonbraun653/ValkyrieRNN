from __future__ import division, print_function, absolute_import

import src.TensorFlowModels as TFModels
import tflearn
import tensorflow as tf
from src.Training import ModelTrainer
from src.Miscellaneous import bcolors
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')   # use('Agg') for saving to file and use('TkAgg') for interactive plot
import matplotlib.pyplot as plt


def train_plot_callback(train_act, train_pred):
    lw = 0.8
    plt.figure(figsize=(32, 18))
    plt.suptitle('Training Data Predictions')

    #plt.subplot(2, 1, 1)
    plt.plot(train_act[:, 0], 'g--', label='Pitch Actual', linewidth=lw)
    plt.plot(train_pred[:, 0], 'r-', label='Pitch Model', linewidth=2 * lw)

    plt.legend()

    # plt.subplot(2, 1, 2)
    # plt.plot(train_act[:, 1], 'b--', label='Roll Actual', linewidth=lw)
    # plt.plot(train_pred[:, 1], 'm-', label='Roll Model', linewidth=2 * lw)
    #
    # plt.legend()


def validate_plot_callback(valid_act, valid_pred):
    lw = 0.8
    plt.figure(figsize=(32, 18))
    plt.suptitle('Validation Data Predictions')

    #plt.subplot(2, 1, 1)
    plt.plot(valid_act[:, 0], 'g--', label='Pitch Actual', linewidth=lw)
    plt.plot(valid_pred[:, 0], 'r-', label='Pitch Model', linewidth=2*lw)
    plt.legend()

    # plt.subplot(2, 1, 2)
    # plt.plot(valid_act[:, 1], 'b--', label='Roll Actual', linewidth=lw)
    # plt.plot(valid_pred[:, 1], 'm-', label='Roll Model', linewidth=2*lw)
    # plt.legend()


if __name__ == "__main__":
    # Global training paths
    train_data_path = 'G:/Projects/ValkyrieRNN/Data/TrainingData/'
    validation_data_path = 'G:/Projects/ValkyrieRNN/Data/ValidationData/'
    prediction_data_path = 'G:/Projects/ValkyrieRNN/Data/PredictionData/'

    results_root = 'G:/Projects/ValkyrieRNN/Simulation/'

    # Generate the model configuration for a large training set
    cfg = TFModels.ModelConfig()
    cfg.model_name = 'pitch_full_ver1'
    cfg.model_type = 'drone_lstm_model_shallow'
    cfg.input_size = 4      # Inputs are motor commands
    cfg.input_depth = 300   # Multiply by 0.002 to get seconds represented
    cfg.output_size = 1     # Outputs are pitch and roll angles
    cfg.batch_len = 128
    cfg.epoch_len = 3
    cfg.neurons_per_layer = 32
    cfg.learning_rate = 0.001
    cfg.layer_dropout = (0.99, 0.99)
    cfg.train_data_len = 25 * 1000
    cfg.data_inversion = False

    cfg.max_cpu_cores = 16
    cfg.max_gpu_mem = 0.6
    cfg.variable_scope = 'euler_pitch'
    cfg.training_device = '/gpu:0'

    config_path = 'C:/git/GitHub/ValkyrieRNN/' + cfg.model_name + '.csv'
    cfg.save(config_path)

    temp = ModelTrainer(config_path=config_path,
                        training_data_path=train_data_path,
                        validation_data_path=validation_data_path,
                        prediction_data_path=prediction_data_path,
                        results_path=results_root)

    in_keys = ['m1CMD', 'm2CMD', 'm3CMD', 'm4CMD']
    out_keys = ['pitch'] #, 'roll']

    temp.train_from_scratch(input_data_keys=in_keys, output_data_keys=out_keys,
                            training_plot_callback=train_plot_callback,
                            validation_plot_callback=validate_plot_callback)

