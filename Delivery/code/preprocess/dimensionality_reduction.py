# coding: utf-8

import numpy
import sklearn.decomposition
import matplotlib.pyplot as plt

import tensorflow as tf

__author__ = 'Joaquim Leitão'
__copyright__ = 'Copyright (c) 2017 Joaquim Leitão'
__email__ = 'jocaleitao93@gmail.com'

allowed_activations = ['sigmoid', 'tanh', 'softmax', 'relu', 'linear']
allowed_noises = [None, 'gaussian', 'mask']
allowed_losses = ['rmse', 'cross-entropy']


class StackedAutoEncoder:
    """
        A deep autoencoder with denoising capability

        Implementation by rajarsheem, available at:
        https://github.com/rajarsheem/libsdae-autoencoder-tensorflow/blob/master/deepautoencoder/stacked_autoencoder.py
    """

    def assertions(self):
        global allowed_activations, allowed_noises, allowed_losses
        assert self.loss in allowed_losses, 'Incorrect loss given'
        assert 'list' in str(
            type(self.dims)), 'dims must be a list even if there is one layer.'
        assert len(self.epoch) == len(self.dims), "No. of epochs must equal to no. of hidden layers"
        assert len(self.activations) == len(
            self.dims), "No. of activations must equal to no. of hidden layers"
        assert all(
            True if x > 0 else False
            for x in self.epoch), "No. of epoch must be atleast 1"
        assert set(self.activations + allowed_activations) == set(
            allowed_activations), "Incorrect activation given."
        assert noise_validator(
            self.noise, allowed_noises), "Incorrect noise given"

    def __init__(self, dims, activations, epoch=1000, noise=None, loss='rmse',
                 lr=0.001, batch_size=100, print_step=50):
        self.print_step = print_step
        self.batch_size = batch_size
        self.lr = lr
        self.loss = loss
        self.activations = activations
        self.noise = noise
        self.epoch = epoch
        self.dims = dims
        self.assertions()
        self.depth = len(dims)
        self.encoder_weights = list()
        self.encoder_biases = list()
        self.decoder_weights = list()
        self.decoder_biases = list()

    def add_noise(self, x):
        if self.noise == 'gaussian':
            n = numpy.random.normal(0, 0.1, (len(x), len(x[0])))
            return x + n
        if 'mask' in self.noise:
            frac = float(self.noise.split('-')[1])
            temp = numpy.copy(x)
            for i in temp:
                n = numpy.random.choice(len(i), round(
                    frac * len(i)), replace=False)
                i[n] = 0
            return temp
        if self.noise == 'sp':
            pass

    def fit(self, x):
        for i in range(self.depth):
            # print('Layer {0}'.format(i + 1))
            if self.noise is None:
                x = self.run(data_x=x, activation=self.activations[i], data_x_=x, hidden_dim=self.dims[i],
                             epoch=self.epoch[i], loss=self.loss, batch_size=self.batch_size, lr=self.lr,
                             print_step=self.print_step)
            else:
                temp = numpy.copy(x)
                x = self.run(data_x=self.add_noise(temp), activation=self.activations[i], data_x_=x,
                             hidden_dim=self.dims[i], epoch=self.epoch[i], loss=self.loss, batch_size=self.batch_size,
                             lr=self.lr, print_step=self.print_step)

    def transform(self, data):
        tf.reset_default_graph()
        sess = tf.Session()

        x = tf.constant(data, dtype=tf.float32)

        for w, b, a in zip(self.encoder_weights, self.encoder_biases, self.activations):
            weight = tf.constant(w, dtype=tf.float32)
            bias = tf.constant(b, dtype=tf.float32)
            layer = tf.matmul(x, weight) + bias
            x = self.activate(layer, a)
        return x.eval(session=sess)

    def fit_transform(self, x):
        self.fit(x)
        return self.transform(x)

    def decode(self, encoded_data):
        tf.reset_default_graph()
        sess = tf.Session()

        curr_decoded = tf.constant(encoded_data, dtype=tf.float32)

        for i in range(len(self.decoder_weights) - 1, -1, -1):
            curr_weights = self.decoder_weights[i]
            curr_bias = self.decoder_biases[i]

            curr_decoded = self.activate(tf.matmul(curr_decoded, curr_weights) + curr_bias, self.activations[i])

        return sess.run(curr_decoded)

    def run(self, data_x, data_x_, hidden_dim, activation, loss, lr,
            print_step, epoch, batch_size=100):

        tf.reset_default_graph()
        sess = tf.Session()

        inumpyut_dim = len(data_x[0])

        x = tf.placeholder(dtype=tf.float32, shape=[None, inumpyut_dim], name='x')
        x_ = tf.placeholder(dtype=tf.float32, shape=[None, inumpyut_dim], name='x_')

        encode = {
            'weights': tf.Variable(tf.truncated_normal([inumpyut_dim, hidden_dim], dtype=tf.float32)),
            'biases': tf.Variable(tf.truncated_normal([hidden_dim], dtype=tf.float32))
        }

        decode = {
            'biases': tf.Variable(tf.truncated_normal([inumpyut_dim], dtype=tf.float32)),
            'weights': tf.transpose(encode['weights'])
        }

        encoded = self.activate(tf.matmul(x, encode['weights']) + encode['biases'], activation)
        decoded = tf.matmul(encoded, decode['weights']) + decode['biases']

        # reconstruction loss
        if loss == 'rmse':
            loss = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(x_, decoded))))
        elif loss == 'cross-entropy':
            loss = -tf.reduce_mean(x_ * tf.log(decoded))

        # train_op = tf.train.AdamOptimizer(lr).minimize(loss)
        train_op = tf.train.GradientDescentOptimizer(lr).minimize(loss)

        sess.run(tf.global_variables_initializer())
        for i in range(epoch):
            b_x, b_x_ = get_batch(data_x, data_x_, batch_size)
            sess.run(train_op, feed_dict={x: b_x, x_: b_x_})
            if (i + 1) % print_step == 0:
                l = sess.run(loss, feed_dict={x: data_x, x_: data_x_})
                print('epoch {0}: global loss = {1}'.format(i, l))
        # debug
        # print('Decoded', sess.run(decoded, feed_dict={x: self.data_x_})[0])
        self.encoder_weights.append(sess.run(encode['weights']))
        self.encoder_biases.append(sess.run(encode['biases']))
        self.decoder_weights.append(sess.run(decode['weights']))
        self.decoder_biases.append(sess.run(decode['biases']))

        return sess.run(encoded, feed_dict={x: data_x_})

    @staticmethod
    def activate(linear, name):
        if name == 'sigmoid':
            return tf.nn.sigmoid(linear, name='encoded')
        elif name == 'softmax':
            return tf.nn.softmax(linear, name='encoded')
        elif name == 'linear':
            return linear
        elif name == 'tanh':
            return tf.nn.tanh(linear, name='encoded')
        elif name == 'relu':
            return tf.nn.relu(linear, name='encoded')

    def assess_reconstruction_performance(self, original, reconstructed):
        """
        Method implemented by @jpleitao
        """
        av_rmse = 0.0
        for i in range(len(original)):
            rmse = 0.0
            for j in range(len(original[i])):
                rmse += (original[i][j] - reconstructed[i][j]) ** 2
            rmse = numpy.sqrt(1 / len(original[i]) * rmse)
            av_rmse += rmse
        av_rmse = av_rmse / len(original)

        print('[SAE]Average RMSE on all data = ' + str(av_rmse))

        for i in range(len(reconstructed)):
            fig = plt.figure()
            plt.plot(reconstructed[i], 'g', label='Reconstructed Data')
            plt.plot(original[i], 'r', label='Original Data')
            plt.title('Day ' + str(i))
            plt.legend()
            fig.savefig('data/images/dimensionality_reduction/stacked_autoencoders/day_' + str(i))
            plt.close(fig)


def get_batch(X, X_, size):
    a = numpy.random.choice(len(X), size, replace=False)
    return X[a], X_[a]


def noise_validator(noise, allowed_noises):
    """Validates the noise provided"""
    try:
        if noise in allowed_noises:
            return True
        elif noise.split('-')[0] == 'mask' and float(noise.split('-')[1]):
            t = float(noise.split('-')[1])
            if 0.0 <= t <= 1.0:
                return True
            else:
                return False
    except:
        return False


def _pca_dimen_reduce(data, percentage_variance=0.8):
    samples, num_features = data.shape
    # First perform PCA with the same number of features as the target number of components -> This is useful when
    # selecting the number of components depending on the percentage of data variance explained by such components
    pca = sklearn.decomposition.PCA(n_components=num_features)
    pca.fit(data)

    variance_ratio = pca.explained_variance_ratio_
    variance_sum = variance_ratio[0]
    i = 0
    while variance_sum < percentage_variance and i < len(variance_ratio):
        i += 1
        variance_sum += variance_ratio[i]
    # Get components
    components = pca.components_[0: i+1]

    # Transform data to the desired number of components - New PCA and fit
    pca = sklearn.decomposition.PCA(n_components=len(components))
    pca.fit(data)
    data_transform = pca.transform(data)

    # Reconstruct and compute RMSE - Average over all days
    data_reconstructed = pca.inverse_transform(data_transform)
    av_rmse = 0.0
    for i in range(len(data)):
        rmse = 0.0
        for j in range(len(data[i])):
            rmse += (data[i][j] - data_reconstructed[i][j])**2
        rmse = numpy.sqrt(1/len(data[i]) * rmse)
        av_rmse += rmse
    av_rmse = av_rmse/len(data)

    print('[PCA]Average RMSE on all data = ' + str(av_rmse) + ' with ' + str(len(components)) + ' components')

    for i in range(len(data_reconstructed)):
        fig = plt.figure()
        plt.plot(data_reconstructed[i], 'g', label='Reconstructed Data')
        plt.plot(data[i], 'r', label='Original Data')
        plt.title('Day ' + str(i))
        plt.legend()
        fig.savefig('data/images/dimensionality_reduction/pca/day_' + str(i))
        plt.close(fig)

    return data_transform, components, pca


def reduce_dimensionality(readings):
    data_transform, eigenvectors, pca = _pca_dimen_reduce(readings, 0.9)

    # The final number of components is given by len(components), so the next step would be to perform dimensionality
    # reduction with a SAE - also plot results there and maybe compare the two with the RMSE on the reconstruction
    num_components = len(eigenvectors)

    dims = [23, 15, 5, num_components]
    activations = ['sigmoid', 'sigmoid', 'sigmoid', 'sigmoid']
    epochs = [2000 for _ in range(len(dims))]
    stacked_autoencoder = StackedAutoEncoder(dims, activations, epochs, batch_size=300)
    transformed = stacked_autoencoder.fit_transform(readings)
    reconstructed = stacked_autoencoder.decode(transformed)

    stacked_autoencoder.assess_reconstruction_performance(readings, reconstructed)

    return data_transform, pca
