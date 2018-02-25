from __future__ import division, print_function, absolute_import

import tflearn
import pandas as pd


class ModelConfig:
    def __init__(self):
        self.input_size = 0
        self.input_depth = 0
        self.output_size = 0
        self.batch_len = 0
        self.epoch_len = 0
        self.neurons_per_layer = 0
        self.layer_dropout = (0, 0)
        self.learning_rate = 0.001

        self.model_name = ''

    def save(self, filename):
        header = "input_size,input_depth,output_size,batch_len,epoch_len," + \
                 "neurons_per_layer,dropout_1,dropout_2,model_name,learning_rate\n"

        with open(filename, 'w') as file:
            file.write(header)

            line = str(self.input_size) + ',' + str(self.input_depth) + ',' + str(self.output_size) + ',' + \
                   str(self.batch_len) + ',' + str(self.epoch_len) + ',' + str(self.neurons_per_layer) + ',' + \
                   str(self.layer_dropout[0]) + ',' + str(self.layer_dropout[1]) + ',' + self.model_name + ',' + \
                   str(self.learning_rate) + '\n'

            file.write(line)

    def load(self, filename):
        config = pd.read_csv(filename)

        self.input_size = int(config['input_size'])
        self.output_size = int(config['output_size'])
        self.input_depth = int(config['input_depth'])
        self.batch_len = int(config['batch_len'])
        self.epoch_len = int(config['epoch_len'])
        self.neurons_per_layer = int(config['neurons_per_layer'])
        self.layer_dropout = (float(config['dropout_1']), float(config['dropout_2']))
        self.learning_rate = float(config['learning_rate'])

        self.model_name = str(config['model_name'][0])


def drone_rnn_model(dim_in, dim_out, past_depth, layer_neurons=128, layer_dropout=1.0,
                    learning_rate=0.001, checkpoint_path='', best_checkpoint_path=''):

    input_layer = tflearn.input_data(shape=[None, dim_in, past_depth])

    layer1 = tflearn.simple_rnn(input_layer,
                                n_units=layer_neurons,
                                activation='relu',
                                return_seq=True,
                                dropout=layer_dropout)

    layer2 = tflearn.simple_rnn(layer1,
                                n_units=layer_neurons,
                                activation='sigmoid',
                                return_seq=True,
                                dropout=layer_dropout)

    layer3 = tflearn.simple_rnn(layer2,
                                n_units=layer_neurons,
                                activation='sigmoid',
                                return_seq=False,
                                dropout=layer_dropout)

    layer4 = tflearn.fully_connected(layer3,
                                     dim_out,
                                     activation='linear')

    output_layer = tflearn.regression(layer4,
                                      optimizer='adam',
                                      loss='mean_square',
                                      learning_rate=learning_rate)

    return tflearn.DNN(output_layer,
                       tensorboard_verbose=3,
                       checkpoint_path=checkpoint_path,
                       best_checkpoint_path=best_checkpoint_path)


def drone_lstm_model(dim_in, dim_out, past_depth, layer_neurons=128, layer_dropout=1.0,
                     learning_rate=0.001, checkpoint_path='', best_checkpoint_path=''):
    input_layer = tflearn.input_data(shape=[None, dim_in, past_depth])

    layer1 = tflearn.lstm(input_layer,
                          n_units=layer_neurons,
                          return_seq=True,
                          dropout=layer_dropout)

    layer2 = tflearn.lstm(layer1,
                          n_units=layer_neurons,
                          return_seq=True,
                          dropout=layer_dropout)

    layer3 = tflearn.lstm(layer2,
                          n_units=layer_neurons,
                          return_seq=False,
                          dropout=layer_dropout)

    layer4 = tflearn.fully_connected(layer3, n_units=dim_out)

    output_layer = tflearn.regression(layer4,
                                      optimizer='adam',
                                      loss='mean_square',
                                      learning_rate=learning_rate)

    return tflearn.DNN(output_layer,
                       tensorboard_verbose=3,
                       checkpoint_path=checkpoint_path,
                       best_checkpoint_path=best_checkpoint_path)

