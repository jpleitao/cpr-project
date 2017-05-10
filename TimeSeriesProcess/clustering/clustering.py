# coding: utf-8

import scipy.cluster.hierarchy
import scipy.spatial.distance
import clustering.distances

import matplotlib.pyplot as plt

__author__ = 'Joaquim Leitão'
__copyright__ = 'Copyright (c) 2017 Joaquim Leitão'
__email__ = 'jocaleitao93@gmail.com'


def hierarchical(readings, time_series=None):
    if time_series is None:
        time_series = True
    elif not isinstance(time_series, bool):
        raise TypeError('Argument <time_series> must be of type <bool>!')

    # FIXME: DEBUG ONLY
    # readings = readings[1:10]

    if time_series:
        # Hierarchical clustering of raw time series data - Use Dynamic Time Warping
        print('Using dynamic time warping')
        distance_metric = clustering.distances.dtw
    else:
        # Hierarchical clustering of transformed time series data - Use Euclidean distance
        print('Using euclidean')
        distance_metric = 'euclidean'

    z = scipy.cluster.hierarchy.linkage(y=readings, method='average', metric=distance_metric)
    plt.figure()
    scipy.cluster.hierarchy.dendrogram(z)


def k_means():
    pass


def clustering_run(readings, data_transform):
    """
    Performs clustering of the time series data. The same clustering methods are intended to be applied to two different
    time series representations: On one of them the raw time series data is intended to be used; on the other the PCA
    dimensionality reduction algorithm was applied to reduce the time series values to a dimensionality capable of
    explaining 80% of the data variance (3 dimensions)
    :param readings: The original time series values
    :param data_transform: The time series values in the reduced (that is, transformed) dimensionality
    """
    # ==================================== Hierarchical Clustering =====================================================
    # Start by performing hierarchical clustering on the original, raw, time series data. For such a data representation
    # an appropriate distance metric must be used, which in this case is the Dynamic Time Warping
    hierarchical(readings)

    # Hierarchical clustering on transformed data (reduced dimensionality)
    hierarchical(data_transform, False)
    plt.show()

    # TODO: Document this!!!
    # Diria que um numero de clusters à volta dos 4;5;6 será um bom indicador (podemos sempre dar um pouco mais...)
