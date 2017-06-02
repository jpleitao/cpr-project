# coding: utf-8

"""
Script that contains the code implemented for the Hierarchical Clustering
"""

import scipy.cluster.hierarchy
import scipy.spatial.distance
import matplotlib.pyplot as plt

import clustering.metrics

__author__ = 'Joaquim Leitão'
__copyright__ = 'Copyright (c) 2017 Joaquim Leitão'
__email__ = 'jocaleitao93@gmail.com'


def hierarchical_clustering(readings, time_series=None):
    if time_series is None:
        time_series = True
    elif not isinstance(time_series, bool):
        raise TypeError('Argument <time_series> must be of type <bool>!')

    if time_series:
        # Hierarchical clustering of raw time series data - Use Dynamic Time Warping
        distance_metric = clustering.metrics.dtw
    else:
        # Hierarchical clustering of transformed time series data - Use Euclidean distance
        distance_metric = 'euclidean'

    z = scipy.cluster.hierarchy.linkage(y=readings, method='average', metric=distance_metric)
    plt.figure()
    scipy.cluster.hierarchy.dendrogram(z)
