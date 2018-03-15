# This follows the tutorial from Python Programming.net here:
# https://pythonprogramming.net/rnn-tensorflow-python-machine-learning-tutorial/
# The purpose in this case is to figure out why the Drone NN isn't even outputting data

import tensorflow as tf
import tflearn
from tensorflow.examples.tutorials.mnist import input_data



mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

hm_epochs = 3
n_classes = 10
batch_size = 128
chunk_size = 28
n_chunks = 28
rnn_size = 128

x = tf.placeholder('float', [None, n_chunks, chunk_size])
y = tf.placeholder('float')

net = tflearn.input_data(shape=[None, n_chunks, chunk_size])
net = tflearn.lstm(net, n_units=rnn_size, return_seq=False)
net = tflearn.fully_connected(net, n_classes, activation='linear')
net = tflearn.regression(net, optimizer='Adam', loss='mean_square', learning_rate=0.001)

model = tflearn.DNN(net, tensorboard_verbose=0)


def train_neural_network():
    for epoch in range(hm_epochs):
        for _ in range(int(mnist.train.num_examples / batch_size)):
            epoch_x, epoch_y = mnist.train.next_batch(batch_size)
            epoch_x = epoch_x.reshape((batch_size, n_chunks, chunk_size))

            model.fit(epoch_x, epoch_y, n_epoch=1)

        print('Epoch', epoch, 'completed out of', hm_epochs)

    print('Accuracy:', model.evaluate(X=mnist.test.images.reshape((-1, n_chunks, chunk_size)), Y=mnist.test.labels))


if __name__ == "__main__":
    train_neural_network()

    # This approach works just as well as the RNN_MNIST_PPN version. The accuracies achieved
    # were within 0.05% of each other, so this DOES work.
