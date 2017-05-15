# coding: utf-8

"""
Stores distance and evaluation metrics used in this work:
    * Regarding Distance metrics, implementations of the following can be found in this file:
        -> Dynamic Time Warping (DTW)
        -> Euclidean Distance.
        -> LB_Keough distance (which is used to compute a lower bound on DTW)
    
    * Regarding Cluster evaluation metrics, implementations of the following can be found:
        -> Silhouette Coefficient        
"""

import numpy

__author__ = 'Joaquim Leitão'
__copyright__ = 'Copyright (c) 2017 Joaquim Leitão'
__email__ = 'jocaleitao93@gmail.com'

# TODO: Implement evaluation metrics and add then to documentation


def silhouette_coefficient(assignments, readings, time_series=None):
    """
    Implementation of the silhouette coefficient metric
    :param assignments: A dictionary containing the assignments of each individual reading to a cluster
    :param readings: The readings values, either a z-normalised time series or a reduced time series (via PCA)
    :param time_series: A boolean value, signalling whether or not the readings are z-normalised or reduced
    :return: The silhouette coefficient for the assigned clusters. The silhouette coefficient is computed as follows:
             For each sample (that is, datum) in readings, the following score is computed:
                 s(i) = (b(i) - a(i)) / max(b(i), a(i))
             where b(i) is the lowest average dissimilarity (distance) of sample i to all samples in other clusters
             and a(i) is the average dissimilarity of sample i with all samples in the same cluster
             
             The value of the silhouette coefficient is computed as an average of all values s(i) for each individual
             reading.
             The average s(i) over all data of the entire dataset is a measure of how appropriately the data has been
             clustered 
    """
    if time_series is None:
        time_series = True
    elif not isinstance(time_series, bool):
        raise TypeError('Argument <time_series> must be of type <bool>!')

    if not isinstance(assignments, dict):
        raise TypeError('Argument <assignments> must be of type <dict>!')

    s = 0.0
    num_s = 0

    # Iterate over each sample
    for ind, i in enumerate(readings):
        # Get cluster of current sample
        sample_cluster = assignments[ind]

        a = 0.0
        num_a = 0
        b = dict()

        # Iterate over all the other samples
        for ind_s, j in enumerate(readings):
            if ind == ind_s:
                continue
            # Compute distance between them
            if time_series:
                dist = dtw(i, j)
            else:
                dist = euclidean(i, j)

            if assignments[ind_s] == sample_cluster:
                # Same cluster - Compute a(i)
                a += dist
                num_a += 1
            else:
                # Different clusters - Compute b(i)
                if assignments[ind_s] not in b.keys():
                    b[assignments[ind_s]] = (0, 0)

                # Update element in dictionary
                b[assignments[ind_s]] = (b[assignments[ind_s]][0] + dist, b[assignments[ind_s]][1] + 1)

        a = a / num_a
        # Compute minimum average distances for other clusters
        min_average = float('inf')
        for k in b.keys():
            curr_average = b[k][0] / b[k][1]

            if curr_average < min_average:
                min_average = curr_average

        s += (min_average - a) / (max(min_average, a))
        num_s += 1

    s = s / num_s
    return s


def euclidean(s1, s2):
    return numpy.sqrt(numpy.sum((s1 - s2) ** 2))


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
