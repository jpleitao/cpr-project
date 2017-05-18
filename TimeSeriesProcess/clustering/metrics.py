# coding: utf-8

"""
Stores distance and evaluation metrics used in this work:
    * Regarding Distance metrics, implementations of the following can be found in this file:
        -> Dynamic Time Warping (DTW)
        -> Euclidean Distance.
        -> LB_Keough distance (which is used to compute a lower bound on DTW)
    
    * Regarding Cluster evaluation metrics, implementations of the following can be found:
        -> Silhouette Coefficient
        -> Calinski-Herabaz Index
        -> Sum of Squared Errors
"""

import numpy
from numba import jit

import sklearn.metrics

import clustering.utils
import clustering.k_means

__author__ = 'Joaquim Leitão'
__copyright__ = 'Copyright (c) 2017 Joaquim Leitão'
__email__ = 'jocaleitao93@gmail.com'

# TODO: Implement evaluation metrics and add then to documentation


def sse(assignments, readings, centroid_type, time_series=None, dba=None):
    if not isinstance(assignments, dict):
        raise TypeError('Argument <assignments> must be of type <dict>!')

    if not isinstance(centroid_type, clustering.k_means.CentroidType):
        raise TypeError('Argument <centroid_type> must be of type <clustering.k_means.CentroidType>!')

    if time_series is None:
        time_series = True
    elif not isinstance(time_series, bool):
        raise TypeError('Argument <time_series> must be of type <bool>!')

    # Compute centroids
    distances, _ = clustering.utils.compute_distances(time_series, readings, None)
    centroids = clustering.utils.compute_centroids(assignments=assignments, readings=readings, distances=distances,
                                                   centroid_type=centroid_type, dba=dba)

    # For each reading get its cluster and sum its distance to the cluster centroid
    # We are going to kind of invert this and start by iterating over the centroids
    sse_sum = 0
    for k in assignments:
        readings_indexes = assignments[k]
        current_centroid = centroids[k]
        for index in readings_indexes:
            # Compute minimum distance to a cluster - That is, compute distance to its centroid
            if time_series:
                dist = dtw(current_centroid, readings[index])
            else:
                dist = euclidean(current_centroid, readings[index])
            sse_sum += (dist**2)
    return sse_sum


def calinski_herabaz_index(assignments, readings, time_series=None):
    if not isinstance(assignments, dict):
        raise TypeError('Argument <assignments> must be of type <dict>!')

    if time_series is None:
        time_series = True
    elif not isinstance(time_series, bool):
        raise TypeError('Argument <time_series> must be of type <bool>!')

    # Get labels for each sample
    reverse_assignments = clustering.utils.reverse_assignments(assignments)
    labels = list()

    for i in range(len(readings)):
        labels.append(reverse_assignments[i])

    return sklearn.metrics.calinski_harabaz_score(readings, labels)


def silhouette_coefficient(assignments, readings, time_series=None):
    """
    Implementation of the silhouette coefficient metric
    :param assignments: A dictionary containing the assignments of each individual reading to a cluster
    :param readings: The readings values, either a z-normalised time series or a reduced time series (via PCA)
    :param time_series: A boolean value, signalling whether or not the readings are z-normalised or reduced
    :return: The silhouette coefficient for the assigned clusters. Implementation from sklearn.metrics.silhouette_score
    """
    if not isinstance(assignments, dict):
        raise TypeError('Argument <assignments> must be of type <dict>!')

    if time_series is None:
        time_series = True
    elif not isinstance(time_series, bool):
        raise TypeError('Argument <time_series> must be of type <bool>!')

    distances, labels = clustering.utils.compute_distances(time_series, readings, assignments)
    return sklearn.metrics.silhouette_score(distances, labels, metric='precomputed')


def euclidean(s1, s2):
    return numpy.sqrt(numpy.sum((s1 - s2) ** 2))


@jit
def dtw(s1, s2, w=None):
    # Calculates dynamic time warping Euclidean distance between two sequences. Option to enforce locality constraint
    # for window w.
    # Implementation highly based on the code available in the GitHub Repository
    # 'time-series-classification-and-clustering' by alexminnaar
    # (https://github.com/alexminnaar/time-series-classification-and-clustering)
    d_t_w = {}

    if w:
        w = max(w, abs(len(s1) - len(s2)))

        for i in range(-1, len(s1)):
            for j in range(-1, len(s2)):
                d_t_w[(i, j)] = float('inf')

    else:
        for i in range(len(s1)):
            d_t_w[(i, -1)] = float('inf')
        for i in range(len(s2)):
            d_t_w[(-1, i)] = float('inf')

    d_t_w[(-1, -1)] = 0

    for i in range(len(s1)):
        if w:
            for j in range(max(0, i - w), min(len(s2), i + w)):
                dist = (s1[i] - s2[j]) ** 2
                d_t_w[(i, j)] = dist + min(d_t_w[(i - 1, j)], d_t_w[(i, j - 1)], d_t_w[(i - 1, j - 1)])
        else:
            for j in range(len(s2)):
                dist = (s1[i] - s2[j]) ** 2
                d_t_w[(i, j)] = dist + min(d_t_w[(i - 1, j)], d_t_w[(i, j - 1)], d_t_w[(i - 1, j - 1)])

    return numpy.sqrt(d_t_w[len(s1) - 1, len(s2) - 1])


@jit
def lb_keogh(s1, s2, r):
    # Calculates LB_Keough lower bound to dynamic time warping. Linear complexity compared to quadratic complexity
    # of dtw
    # Implementation highly based on the code available in the GitHub Repository
    # 'time-series-classification-and-clustering' by alexminnaar
    # (https://github.com/alexminnaar/time-series-classification-and-clustering)
    l_b_sum = 0

    for ind, i in enumerate(s1):

        lower_bound = min(s2[(ind - r if ind - r >= 0 else 0):(ind + r)])
        upper_bound = max(s2[(ind - r if ind - r >= 0 else 0):(ind + r)])

        if i > upper_bound:
            l_b_sum = l_b_sum + (i - upper_bound) ** 2
        elif i < lower_bound:
            l_b_sum = l_b_sum + (i - lower_bound) ** 2

    return numpy.sqrt(l_b_sum)
