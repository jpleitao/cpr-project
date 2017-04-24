# coding: utf-8

import numpy
import sklearn.decomposition

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

    # Transform data
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

    print('[PCA]Average RMSE on all data= ' + str(av_rmse))

    return data_transform, components


def reduce_dimensionality(time, readings):
    data_transform, eigenvectors = _pca_dimen_reduce(readings, 0.8)

    # The final number of components is given by len(components), so the next step would be to perform dimensionality
    # reduction with a SAE - also plot results there and maybe compare the two with the RMSE on the reconstruction
    num_components = eigenvectors
