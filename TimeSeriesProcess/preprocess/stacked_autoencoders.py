# coding: utf-8

import math

import numpy
import tensorflow as tf

__author__ = 'Joaquim Leitão'
__copyright__ = 'Copyright (c) 2017 Joaquim Leitão'
__email__ = 'jocaleitao93@gmail.com'

_REPORT_TRAINING = False


# FIXME: Ditch this implementation and try the following:
# https://github.com/rajarsheem/libsdae-autoencoder-tensorflow/blob/master/deepautoencoder/stacked_autoencoder.py
# It's already a denoising stacked autoencoder
# Try to see if we can reconstruct the input with a minimum error, if not just forget about it and use PCA!!

class _DenoisedAutoencoder(object):
    """
    Simple implementation of a Denoised Autoencoder
    """

    cost_functions = ['mean_square', 'cross_entropy']
    train_methods = ['gradient_descent', 'adam']

    def __init__(self, input_size, input_type, hidden_layer_size, cost=None):
        """
        Class constructor
        :param input_size: 
        :param input_type:
        :param hidden_layer_size: 
        :param cost:
        """
        # Check arguments' type
        check_input_size(input_size)
        check_input_type(input_type)
        check_layer_size(hidden_layer_size)
        cost = check_cost(cost)

        # Create placeholder for input
        self._inputs = tf.placeholder(input_type, [None, input_size])

        # Create tensorflow variable for weights - Randomly initialize them with values in interval:
        # [-1/sqrt(n) , 1/sqrt(n)] - FIXME: Maybe try with tf.random_normal
        min_val = -1.0 / math.sqrt(input_size)
        max_val = 1.0 / math.sqrt(input_size)
        self._weights_h = tf.Variable(tf.random_uniform([input_size, hidden_layer_size], min_val, max_val))

        # Create tensorflow variable for bias - Initialize to zero - FIXME: Maybe try with tf.random_normal
        self._bias_h = tf.Variable(tf.zeros([hidden_layer_size]))

        # Create hidden layer, aka encoder - FIXME: Maybe try other activation functions such as sigmoid
        # self._encoder = tf.nn.tanh(tf.matmul(self._inputs, self._weights_h) + self._bias_h)
        self._encoder = self.encoder_expr(self._inputs, self._weights_h, self._bias_h)

        # Create the output layer, aka decoder - FIXME: Use the same activation function as in the hidden layer
        # Using tied weights method - Weights in the hidden layer are transposed of weights in output layer!
        self._weights_o = tf.transpose(self._weights_h)
        self._bias_o = tf.Variable(tf.zeros([input_size]))  # FIXME: Activation function
        # self._decoder = tf.nn.tanh(tf.matmul(self._encoder, self._weights_o) + self._bias_o)
        self._decoder = self.decoder_expr(self._encoder, self._weights_o, self._bias_o)

        # Create cost function
        self._cost_function(cost.lower())

    @staticmethod
    def encoder_expr(dataset, weights, bias):
        if isinstance(dataset, (numpy.ndarray, numpy.generic)):
            dataset = dataset.astype(numpy.float32)
        return tf.nn.sigmoid(tf.matmul(dataset, weights) + bias)

    @staticmethod
    def decoder_expr(dataset, weights, bias):
        return tf.nn.sigmoid(tf.matmul(dataset, weights) + bias)

    def _cost_function(self, cost):
        if cost == 'mean_square':
            self._cost = tf.sqrt(tf.reduce_mean(tf.square(self._inputs - self._decoder)))
        elif cost == 'cross_entropy':
            self._cost = - tf.reduce_sum(self._inputs * tf.log(self._decoder))
        else:
            # Shouldn't happen
            raise Exception('Unexpected value "' + str(cost) + '" for variable <cost>!')

    def train(self, sess, dataset, epochs=None, learning_rate=None, method=None):
        # TODO: dataset must be a list or numpy list!
        epochs = check_epochs(epochs)
        method = check_method(method)
        learning_rate = check_learning_rate(learning_rate)

        # Create optimizer
        if method.lower() == 'gradient_descent':
            optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        elif method.lower == 'adam':
            optimizer = tf.train.AdamOptimizer(learning_rate)
        else:
            raise Exception('Invalid value for the training method!')

        optimizer = optimizer.minimize(self._cost)

        for i in range(epochs):
            sess.run(optimizer, feed_dict={self._inputs: dataset})

            if _REPORT_TRAINING and (i % 100 == 0):
                # Report the training cost!
                print(str(i) + " cost " + str(sess.run(self._cost, feed_dict={self._inputs: dataset})))

    def encode_dataset(self, dataset, sess=None):
        if sess is None:
            return self.encoder_expr(dataset, self._weights_h, self._bias_h)
        return sess.run(self._encoder, feed_dict={self._inputs: dataset})

    def decode_dataset(self, dataset, sess=None):
        if sess is None:
            return self.decoder_expr(dataset, self._weights_o, self._bias_o)
        return sess.run(self._decoder, feed_dict={self._decoder: dataset})


class StackedDenoisedAutoencoder(object):
    """
    Simple Implementation of a Stacked Denoised Autoencoder
    """

    def __init__(self, input_size, input_type, hidden_layers_sizes, learning_rate=None, epochs=None, train_method=None,
                 cost=None):
        self._hidden_layers_sizes = check_layer_sizes(hidden_layers_sizes)
        self._learning_rate = check_learning_rate(learning_rate)
        self._epochs = check_epochs(epochs)
        self._train_method = train_method
        cost = check_cost(cost)

        curr_size = input_size

        # Create list of encoders
        self._autoencoders = list()
        for i in range(self._hidden_layers_sizes.size):
            self._autoencoders.append(_DenoisedAutoencoder(curr_size, input_type, self._hidden_layers_sizes[i], cost))
            curr_size = self._hidden_layers_sizes[i]

        # Global variables initializer task
        self._init = tf.global_variables_initializer()

        # Get Tensorflow session
        self._sess = tf.Session()

    def train(self, dataset):
        """
        Train the autoencoders
        :param dataset: The training dataset
        """
        # Initialize global variables
        self._sess.run(self._init)

        # Start training autoencoders
        encoder_input = dataset
        for i in range(len(self._autoencoders)):
            self._autoencoders[i].train(self._sess, encoder_input, self._epochs, self._learning_rate)

            # Get encoder output, which will be the training input of the next encoder
            encoder_input = self._autoencoders[i].encode_dataset(encoder_input, self._sess)

    def evaluate_encoder_performance(self, data):
        encoded_data = self.encode_data(data)
        data_reconstructed = self._decode_data(encoded_data)

        encoded_data = self._sess.run(encoded_data)
        data_reconstructed = self._sess.run(data_reconstructed)

        print(data_reconstructed[0])

        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(data_reconstructed[0], 'g')
        plt.plot(data[0], 'r')
        plt.show()

    def encode_data(self, data):
        """
        Encodes the given data, using the trained encoders to process it 
        :param data: The data to be encoded
        :return: 
        """
        encoder_input = data
        for i in range(len(self._autoencoders)):
            encoder_input = self._autoencoders[i].encode_dataset(encoder_input)
        return encoder_input

    def _decode_data(self, encoded_data):
        decoder_input = encoded_data

        for i in range(len(self._autoencoders)-1, -1, -1):
            decoder_input = self._autoencoders[i].decode_dataset(decoder_input)
        return decoder_input

    def close_session(self):
        self._sess.close()
        self._sess = None


def check_input_size(input_size):
    if not isinstance(input_size, (int, numpy.int, numpy.int8, numpy.int16, numpy.int32, numpy.int64)):
        raise TypeError('Invalid type for argument <input_size>!')
    elif input_size <= 0:
        raise ValueError('Invalid value for argument <input_size>!')


def check_input_type(input_type):
    if not isinstance(input_type, str):
        raise TypeError('Invalid type for argument <input_type>!')


def check_layer_sizes(hidden_layers_sizes):
    if isinstance(hidden_layers_sizes, (numpy.ndarray, numpy.generic)):
        # Check if 1D array
        if hidden_layers_sizes.ndim > 1:
            raise ValueError('<hidden_layers_sizes> must be 1-D array!')
    elif isinstance(hidden_layers_sizes, list):
        # Also check dimensions
        hidden_layers_sizes = numpy.array(hidden_layers_sizes)
        if hidden_layers_sizes.ndim > 1:
            raise ValueError('<hidden_layers_sizes> must be 1-D array!')
    else:
        raise TypeError('Invalid type for argument <hidden_layers_sizes>!')

    return hidden_layers_sizes


def check_layer_size(hidden_layer_size):
    if not isinstance(hidden_layer_size,
                      (int, numpy.int, numpy.int8, numpy.int16, numpy.int32, numpy.int64)):
        raise TypeError('Invalid type for argument <hidden_layer_size>!')
    elif hidden_layer_size <= 0:
        raise ValueError('Invalid size for argument <hidden_layer_size>!')


def check_learning_rate(learning_rate):
    # TODO: Proper type checking here!
    try:
        learning_rate = float(learning_rate)
    except Exception:
        learning_rate = 0.01
    return learning_rate


def check_cost(cost):
    if cost is None:
        cost = 'mean_square'
    elif not isinstance(cost, str):
        raise TypeError('Invalid type for argument <cost>!')
    elif cost.lower() not in _DenoisedAutoencoder.cost_functions:
        raise ValueError('Invalid value for argument <cost>!')
    return cost


def check_method(method):
    if method is None:
        method = 'gradient_descent'
    elif not isinstance(method, str):
        raise TypeError('Invalid type for argument <method>!')
    elif method.lower() not in _DenoisedAutoencoder.train_methods:
        raise ValueError('Invalid value for argument <method>!')
    return method


def check_epochs(epochs):
    if epochs is None:
        epochs = 2000
    elif not isinstance(epochs, int):
        raise TypeError('Invalid type for argument <epochs>!')
    elif epochs <= 0:
        raise ValueError('Invalid number of epochs for training!')
    return epochs
