# coding: utf-8

import math

import numpy
import tensorflow as tf

__author__ = 'Joaquim Leitão'
__copyright__ = 'Copyright (c) 2017 Joaquim Leitão'
__email__ = 'jocaleitao93@gmail.com'

# TODO: Check how to share variables on tensorflow, how exactly it works! Try with a toy network first!!!


class DenoisedAutoencoder(object):
    """
    Simple implementation of a Denoised Autoencoder
    """

    _AVAILABLE_COST_FUNCTIONS = ['mean_square', 'cross_entropy']
    _AVAILABLE_TRAIN_METHODS = ['gradient_descent', 'adam']

    # TODO: turn this into densoising autoencoder!

    def __init__(self, input_size, input_type, hidden_layer_size, cost=None):
        """
        Class constructor
        :param input_size: 
        :param input_type:
        :param hidden_layer_size: 
        """
        # Check arguments' type
        if not isinstance(input_size, int):
            raise TypeError('Invalid type for argument <input_size>!')
        elif input_size <= 0:
            raise ValueError('Invalid size for autoencoder input!')
        if not isinstance(input_type, str):
            raise TypeError('Invalid type for argument <input_type>!')
        if not isinstance(hidden_layer_size, int):
            raise TypeError('Invalid type for argument <hidden_layer_size>!')
        elif hidden_layer_size <= 0:
            raise ValueError('Invalid size for autoencoder hidden layer!')
        if cost is None:
            cost = 'mean_square'
        elif not isinstance(cost, str):
            raise TypeError('Invalid type for argument <cost>!')
        elif cost.lower() not in DenoisedAutoencoder._AVAILABLE_COST_FUNCTIONS:
            raise ValueError('Invalid value for argument <cost>!')

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
        self._encoder = tf.nn.tanh(tf.matmul(self._inputs, self._weights_h) + self._bias_h)

        # Create the output layer, aka decoder - FIXME: Use the same activation function as in the hidden layer
        # Using tied weights method - Weights in the hidden layer are transposed of weights in output layer!
        self._weights_o = tf.transpose(self._weights_h)
        self._bias_o = tf.Variable(tf.zeros([input_size]))  # FIXME: Activation function
        self._decoder = tf.nn.tanh(tf.matmul(self._encoder, self._weights_o) + self._bias_o)

        # Create cost function
        self._cost_function(cost.lower())

    def _cost_function(self, cost):
        if cost == 'mean_square':
            self._cost = tf.sqrt(tf.reduce_mean(tf.square(self._inputs - self._decoder)))
        elif cost == 'cross_entropy':
            self._cost = - tf.reduce_sum(self._inputs * tf.log(self._decoder))
        else:
            # Shouldn't happen
            raise Exception('Unexpected value "' + str(cost) + '" for variable <cost>!')

    @property
    def cost(self):
        return self._cost

    @property
    def encoder(self):
        return self._encoder

    def train(self, sess, dataset, epochs=None, learning_rate=None, method=None):
        # TODO: dataset must be a list or numpy list!

        if epochs is None:
            epochs = 2000
        elif not isinstance(epochs, int):
            raise TypeError('Invalid type for argument <epochs>!')
        elif epochs <= 0:
            raise ValueError('Invalid number of epochs for training!')

        if method is None:
            method = 'gradient_descent'
        elif not isinstance(method, str):
            raise TypeError('Invalid type for argument <method>!')
        elif method.lower() not in DenoisedAutoencoder._AVAILABLE_TRAIN_METHODS:
            raise ValueError('Invalid value for argument <method>!')

        # TODO: Proper type checking here!
        try:
            learning_rate = float(learning_rate)
        except Exception:
            learning_rate = 0.01

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

            # TODO: Maybe compute error/performance in training!
            # See https://gist.github.com/vinhkhuc/e53a70f9e5c3f55852b0


class StackedDenoisedAutoencoder(object):
    """
    Simple Implementation of a Stacked Denoised Autoencoder
    """

    # TODO: Implement this!

    def __init__(self, input_size, hidden_layers_sizes, learning_rate, epochs):
        """
        Class constructor
        """


        # Create list of encoders


if __name__ == '__main__':


