# coding: utf-8

import numpy
import sklearn.decomposition

import preprocess

__author__ = 'Joaquim Leitão'
__copyright__ = 'Copyright (c) 2017 Joaquim Leitão'
__email__ = 'jocaleitao93@gmail.com'


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

    import matplotlib.pyplot as plt

    plt.figure()
    plt.plot(data_reconstructed[0], 'g')
    plt.plot(data[0], 'r')

    return data_transform, components


def reduce_dimensionality(readings):
    data_transform, eigenvectors = _pca_dimen_reduce(readings, 0.9)

    # The final number of components is given by len(components), so the next step would be to perform dimensionality
    # reduction with a SAE - also plot results there and maybe compare the two with the RMSE on the reconstruction
    num_components = len(eigenvectors)
    print('Got ' + str(num_components) + ' components')

    stacked_autoencoder = preprocess.StackedDenoisedAutoencoder(readings.shape[1],
                                                                numpy.asscalar(readings[0, 0]).__class__.__name__,
                                                                [num_components],
                                                                0.01,
                                                                2000,
                                                                'gradient_descent',
                                                                'mean_square')
    print('Started training stacked autoencoder')
    stacked_autoencoder.train(readings)
    print('Finished training stacked autoencoder')
    stacked_autoencoder.evaluate_encoder_performance(readings)

    stacked_autoencoder.close_session()
