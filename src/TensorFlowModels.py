from __future__ import division, print_function, absolute_import

import tflearn
import pandas as pd


class ModelConfig:
    def __init__(self):
        self._config_data = pd.DataFrame()

        self._input_size_key = 'input_size'
        self._input_depth_key = 'input_depth'
        self._output_size_key = 'output_size'
        self._batch_len_key = 'batch_len'
        self._epoch_len_key = 'epoch_len'
        self._neurons_per_layer_key = 'neurons_per_layer'
        self._layer_dropout1_key = 'layer_dropout_1'
        self._layer_dropout2_key = 'layer_dropout_2'
        self._learning_rate_key = 'learning_rate'
        self._model_name_key = 'model_name'
        self._model_type_key = 'model_type'
        self._epoch_chkpt_path_key = 'epoch_chkpt_path'
        self._best_chkpt_path_key = 'best_chkpt_path'
        self._last_chkpt_path_key = 'last_chkpt_path'
        self._train_data_path_key = 'train_data_path'
        self._train_data_len_key = 'train_data_len'
        self._image_data_path_key = 'img_data_path'
        self._data_inversion_key = 'data_inversion'
        self._variable_scope_key = 'var_scope'
        self._max_cpu_cores_key = 'max_cpu_cores'
        self._max_gpu_mem_key = 'max_gpu_mem'
        self._training_device_key = 'train_dev_key'
        self._input_data_key = 'input_key'
        self._output_data_key = 'output_key'


    def save(self, file_name):
        self._config_data.to_csv(file_name, sep=',', encoding='utf-8')

    def load(self, file_name):
        self._config_data = pd.read_csv(file_name)

    def append_new_column(self, key='', value=None):
        assert(isinstance(key, str))

        if key in self._config_data.keys():
            raise ValueError("Key already exists. Did you mean overwrite_column?")

        self._config_data[key] = pd.Series([value])

    def overwrite_column(self, key='', value=None):
        assert (isinstance(key, str))
        self._config_data[key] = pd.Series([value])

    def read_column(self, key=''):
        assert (isinstance(key, str))
        if key not in self._config_data.keys():
            raise ValueError("Key doesn't exist?")
        return self._config_data[key][0]

    @property
    def input_keys(self):
        return self._config_data[self._input_data_key][0]

    @input_keys.setter
    def input_keys(self, value):
        self._config_data[self._input_data_key] = pd.Series([value])

    @property
    def output_keys(self):
        return self._config_data[self._output_data_key][0]

    @output_keys.setter
    def output_keys(self, value):
        self._config_data[self._output_data_key] = pd.Series([value])

    @property
    def training_device(self):
        return self._config_data[self._training_device_key][0]

    @training_device.setter
    def training_device(self, value):
        self._config_data[self._training_device_key] = pd.Series([value])

    @property
    def input_size(self):
        return self._config_data[self._input_size_key][0]

    @input_size.setter
    def input_size(self, value):
        self._config_data[self._input_size_key] = pd.Series([value])

    @property
    def input_depth(self):
        return self._config_data[self._input_depth_key][0]

    @input_depth.setter
    def input_depth(self, value):
        self._config_data[self._input_depth_key] = pd.Series([value])

    @property
    def output_size(self):
        return self._config_data[self._output_size_key][0]

    @output_size.setter
    def output_size(self, value):
        self._config_data[self._output_size_key] = pd.Series([value])

    @property
    def batch_len(self):
        return self._config_data[self._batch_len_key][0]

    @batch_len.setter
    def batch_len(self, value):
        self._config_data[self._batch_len_key] = pd.Series([value])

    @property
    def epoch_len(self):
        return self._config_data[self._epoch_len_key][0]

    @epoch_len.setter
    def epoch_len(self, value):
        self._config_data[self._epoch_len_key] = pd.Series([value])

    @property
    def neurons_per_layer(self):
        return self._config_data[self._neurons_per_layer_key][0]

    @neurons_per_layer.setter
    def neurons_per_layer(self, value):
        self._config_data[self._neurons_per_layer_key] = pd.Series([value])

    @property
    def layer_dropout(self):
        _x = self._config_data[self._layer_dropout1_key][0]
        _y = self._config_data[self._layer_dropout2_key][0]
        return tuple([_x, _y])

    @layer_dropout.setter
    def layer_dropout(self, value):
        assert(isinstance(value, tuple))
        self._config_data[self._layer_dropout1_key] = pd.Series([value[0]])
        self._config_data[self._layer_dropout2_key] = pd.Series([value[1]])

    @property
    def learning_rate(self):
        return self._config_data[self._learning_rate_key][0]

    @learning_rate.setter
    def learning_rate(self, value):
        self._config_data[self._learning_rate_key] = pd.Series([value])

    @property
    def model_name(self):
        return self._config_data[self._model_name_key][0]

    @model_name.setter
    def model_name(self, value):
        self._config_data[self._model_name_key] = pd.Series([value])

    @property
    def epoch_chkpt_path(self):
        return self._config_data[self._epoch_chkpt_path_key][0]

    @epoch_chkpt_path.setter
    def epoch_chkpt_path(self, value):
        self._config_data[self._epoch_chkpt_path_key] = pd.Series([value])

    @property
    def best_chkpt_path(self):
        return self._config_data[self._best_chkpt_path_key][0]

    @best_chkpt_path.setter
    def best_chkpt_path(self, value):
        self._config_data[self._best_chkpt_path_key] = pd.Series([value])

    @property
    def last_chkpt_path(self):
        return self._config_data[self._last_chkpt_path_key][0]

    @last_chkpt_path.setter
    def last_chkpt_path(self, value):
        self._config_data[self._last_chkpt_path_key] = pd.Series([value])

    @property
    def train_data_path(self):
        return self._config_data[self._train_data_path_key][0]

    @train_data_path.setter
    def train_data_path(self, value):
        self._config_data[self._train_data_path_key] = pd.Series([value])

    @property
    def train_data_len(self):
        return self._config_data[self. _train_data_len_key][0]

    @train_data_len.setter
    def train_data_len(self, value):
        self._config_data[self. _train_data_len_key] = pd.Series([value])

    @property
    def model_type(self):
        return self._config_data[self._model_type_key][0]

    @model_type.setter
    def model_type(self, value):
        self._config_data[self._model_type_key] = pd.Series([value])

    @property
    def image_data_path(self):
        return self._config_data[self._image_data_path_key][0]

    @image_data_path.setter
    def image_data_path(self, value):
        self._config_data[self._image_data_path_key] = pd.Series([value])

    @property
    def data_inversion(self):
        return self._config_data[self._data_inversion_key][0]

    @data_inversion.setter
    def data_inversion(self, value):
        self._config_data[self._data_inversion_key] = pd.Series([value])

    @property
    def variable_scope(self):
        return self._config_data[self._variable_scope_key][0]

    @variable_scope.setter
    def variable_scope(self, value):
        self._config_data[self._variable_scope_key] = pd.Series([value])

    @property
    def max_cpu_cores(self):
        return self._config_data[self._max_cpu_cores_key][0]

    @max_cpu_cores.setter
    def max_cpu_cores(self, value):
        self._config_data[self._max_cpu_cores_key] = pd.Series([value])

    @property
    def max_gpu_mem(self):
        return self._config_data[self._max_gpu_mem_key][0]

    @max_gpu_mem.setter
    def max_gpu_mem(self, value):
        self._config_data[self._max_gpu_mem_key] = pd.Series([value])


def drone_rnn_model(shape, dim_in, dim_out, past_depth, layer_neurons=128, layer_dropout=1.0,
                    learning_rate=0.001, checkpoint_path='', best_checkpoint_path='',
                    log_dir='/tmp/tflearn_logs/'):
    input_layer = tflearn.input_data(shape=shape)

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
                       best_checkpoint_path=best_checkpoint_path,
                       tensorboard_dir=log_dir)


def drone_lstm_model_deep(shape, dim_in, dim_out, past_depth, layer_neurons=128, layer_dropout=1.0,
                          learning_rate=0.001, checkpoint_path='', best_checkpoint_path='',
                          log_dir='/tmp/tflearn_logs/'):
    input_layer = tflearn.input_data(shape=shape)

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
                          return_seq=True,
                          dropout=layer_dropout)

    layer4 = tflearn.lstm(layer3,
                          n_units=layer_neurons,
                          return_seq=True,
                          dropout=layer_dropout)

    layer5 = tflearn.lstm(layer4,
                          n_units=layer_neurons,
                          return_seq=False,
                          dropout=layer_dropout)

    layer6 = tflearn.fully_connected(layer5, n_units=dim_out)

    output_layer = tflearn.regression(layer6,
                                      optimizer='adam',
                                      loss='mean_square',
                                      learning_rate=learning_rate)

    return tflearn.DNN(output_layer,
                       tensorboard_verbose=3,
                       checkpoint_path=checkpoint_path,
                       best_checkpoint_path=best_checkpoint_path,
                       tensorboard_dir=log_dir)


def drone_lstm_deeply_connected(shape, dim_in, dim_out, past_depth, layer_neurons=128, layer_dropout=1.0,
                                learning_rate=0.001, checkpoint_path='', best_checkpoint_path='',
                                log_dir='/tmp/tflearn_logs/'):
    input_layer = tflearn.input_data(shape=shape)

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

    layer5 = tflearn.fully_connected(layer4, n_units=dim_out)

    output_layer = tflearn.regression(layer5,
                                      optimizer='adam',
                                      loss='mean_square',
                                      learning_rate=learning_rate)

    return tflearn.DNN(output_layer,
                       tensorboard_verbose=3,
                       checkpoint_path=checkpoint_path,
                       best_checkpoint_path=best_checkpoint_path,
                       tensorboard_dir=log_dir)


def drone_lstm_sandbox1(shape, dim_in, dim_out, past_depth, layer_neurons=128, layer_dropout=1.0,
                        learning_rate=0.001, checkpoint_path='', best_checkpoint_path='',
                        log_dir='/tmp/tflearn_logs/'):
    input_layer = tflearn.input_data(shape=shape,
                                     data_preprocessing=None,
                                     name='InputData')

    layer1 = tflearn.lstm(incoming=input_layer, n_units=layer_neurons, return_seq=True, dropout=layer_dropout,
                          activation='tanh',
                          inner_activation='sigmoid',
                          weights_init='uniform',
                          name='LSTM_Layer1')

    layer2 = tflearn.lstm(incoming=layer1, n_units=layer_neurons, return_seq=False, dropout=layer_dropout,
                          activation='tanh',
                          inner_activation='sigmoid',
                          weights_init='uniform',
                          name='LSTM_Layer2')

    layer3 = tflearn.fully_connected(incoming=layer2, n_units=layer_neurons, activation='tanh')
    layer4 = tflearn.fully_connected(incoming=layer3, n_units=dim_out, activation='linear')

    output_layer = tflearn.regression(layer4, optimizer='adam', loss='categorical_crossentropy',
                                      learning_rate=learning_rate)

    return tflearn.DNN(network=output_layer,
                       tensorboard_verbose=3,
                       checkpoint_path=checkpoint_path,
                       best_checkpoint_path=best_checkpoint_path,
                       tensorboard_dir=log_dir)


def drone_gru_sandbox1(shape, dim_in, dim_out, past_depth, layer_neurons=128, layer_dropout=1.0,
                       learning_rate=0.001, checkpoint_path='', best_checkpoint_path='',
                       log_dir='/tmp/tflearn_logs/'):
    input_layer = tflearn.input_data(shape=shape,
                                     data_preprocessing=None,
                                     name='InputData')

if __name__ == '__main__':
    pass