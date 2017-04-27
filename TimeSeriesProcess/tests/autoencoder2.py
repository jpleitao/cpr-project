# coding: utf-8

import math

import tensorflow.examples.tutorials.mnist
import tensorflow as tf


__author__ = 'Joaquim Leitão'
__copyright__ = 'Copyright (c) 2017 Joaquim Leitão'
__email__ = 'jocaleitao93@gmail.com'

# TODO: Move implementation to classes!
# TODO: Convert to denoising autoencoders!
# TODO: Compute the output of the stacked autoencoders for a given input
# TODO: "Play" with the activation functions


def create_autoencoder(x, hidden_layer_size):
    # ========================================== Build the encoder =====================================================
    hidden_layer_input = x
    input_dim = int(hidden_layer_input.get_shape()[1])

    # Initialize W using random values in interval [-1/sqrt(n) , 1/sqrt(n)] - FIXME: Maybe try with tf.random_normal
    min_val = -1.0 / math.sqrt(input_dim)
    max_val = 1.0 / math.sqrt(input_dim)
    W = tf.Variable(tf.random_uniform([input_dim, hidden_layer_size], min_val, max_val))

    # Initialize b to zero - FIXME: Maybe try with tf.random_normal
    b = tf.Variable(tf.zeros([hidden_layer_size]))

    # FIXME: Maybe try other activation functions such as sigmoid
    encoded_x = tf.nn.tanh(tf.matmul(hidden_layer_input, W) + b)

    # ========================================== Build the decoder =====================================================
    # Using tied weights: The encode and decode weight matrices are simply transposes of each other.
    W = tf.transpose(W)
    b = tf.Variable(tf.zeros([input_dim]))

    decoded_x = tf.nn.tanh(tf.matmul(encoded_x, W) + b)

    return {
        'encoded': encoded_x,
        'decoded': decoded_x,
        'weights': W,
        'bias': b,
        'cost': tf.sqrt(tf.reduce_mean(tf.square(x - decoded_x)))
    }


def merge_autoencoders(first_autoencoder, second_autoencoder):
    print(first_autoencoder['encoded'])

    print(second_autoencoder['encoded'])


def simple_test():
    mnist = tensorflow.examples.tutorials.mnist.input_data.read_data_sets("MNIST_data/", one_hot=True)

    learning_rate = 0.01
    batch_size = 100
    epochs = 2000

    # First autoencoder
    x_fist = tf.placeholder("float", [None, 784])
    first_autoencoder = create_autoencoder(x_fist, 100)

    # Second autoencoder
    x_second = tf.placeholder("float", [None, 100])
    second_autoencoder = create_autoencoder(x_second, 10)

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)

        # =========================================== Train first encoder ==============================================
        train_step_first = tf.train.GradientDescentOptimizer(learning_rate).minimize(first_autoencoder['cost'])

        print('====================================== Train first encoder ===========================================')

        for i in range(epochs):
            # Get a batch
            batch, _ = mnist.train.next_batch(batch_size)

            # Run train step
            sess.run(train_step_first, feed_dict={x_fist: batch})
            if i % 100 == 0:
                print(str(i) + " cost " + str(sess.run(first_autoencoder['cost'], feed_dict={x_fist: batch})))

            # print(" original" + str(batch[0]))
            # print(" decoded" + str(sess.run(autoencoder['decoded'], feed_dict={x: batch})))

        # =========================================== Train second encoder =============================================
        train_step_second = tf.train.GradientDescentOptimizer(learning_rate).minimize(second_autoencoder['cost'])

        print('====================================== Train second encoder ===========================================')

        for i in range(epochs):
            # Get a batch
            batch, _ = mnist.train.next_batch(batch_size)

            # Run the first autoencoder on this dataset to get the input and output
            second_inputs = sess.run(first_autoencoder['encoded'], feed_dict={x_fist: batch})

            # Run train step
            sess.run(train_step_second, feed_dict={x_second: second_inputs})

            if i % 100 == 0:
                print(str(i) + " cost " + str(sess.run(second_autoencoder['cost'],
                                                       feed_dict={x_second: second_inputs})))

        # ============================================ Merge Autoencoders ==============================================
        merge_autoencoders(first_autoencoder, second_autoencoder)


if __name__ == '__main__':
    simple_test()
