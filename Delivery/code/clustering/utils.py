# coding: utf-8

"""
Contains utility functions used throughout the clustering package
"""

import datetime
import numpy

import clustering.dba
import clustering.metrics
import clustering.k_means

__author__ = 'Joaquim Leitão'
__copyright__ = 'Copyright (c) 2017 Joaquim Leitão'
__email__ = 'jocaleitao93@gmail.com'


def _compute_centroids_medoid(assignments, readings, distances, centroids=None):
    if centroids is None:
        centroids = dict()
    for key in assignments:
        min_index = None
        min_avg = float('inf')
        cur_elements = assignments[key]

        for my_i in range(len(cur_elements)):
            avg = 0
            for my_j in range(len(cur_elements)):
                if my_j != my_i:
                    avg += distances[my_i][my_j]
            avg = avg / (len(distances) - 1)

            if avg < min_avg:
                min_avg = avg
                min_index = my_i

        centroids[key] = readings[min_index]
    return centroids


def _compute_centroids_dba(assignments, readings, dba, centroids=None):
    if centroids is None:
        centroids = dict()
    for key in assignments:
        cluster_time_series_list = list()
        for k in assignments[key]:
            cluster_time_series_list.append(readings[k])
        cluster_time_series_list = numpy.array(cluster_time_series_list)
        len_series = len(cluster_time_series_list[0])

        cluster_time_series_list = clustering.dba.ts_to_dba_list(cluster_time_series_list)
        result = dba.compute_average(tseries=cluster_time_series_list, nstarts=5, dba_length=len_series)

        centroids[key] = result.reshape((1, len(result)))[0]
    return centroids


def _compute_centroids_average(assignments, readings, centroids=None):
    if centroids is None:
        centroids = dict()

    # Compute centroid as average of all time series in the cluster
    for key in assignments:
        clust_sum = 0
        for k in assignments[key]:
            clust_sum = clust_sum + readings[k]
        centroids[key] = [m / len(assignments[key]) for m in clust_sum]
    return centroids


def compute_centroids(assignments, readings, distances, centroid_type, centroids=None, dba=None):
    if centroid_type == clustering.k_means.CentroidType.MEDOID:
        return _compute_centroids_medoid(assignments, readings, distances, centroids)
    elif centroid_type == clustering.k_means.CentroidType.AVERAGE:
        return _compute_centroids_average(assignments, readings, centroids)

    return _compute_centroids_dba(assignments, readings, dba, centroids)


def compute_distances(time_series, readings, assignments):
    distances = numpy.zeros((len(readings), len(readings)))

    if assignments is None:
        labels = None
    else:
        labels = numpy.zeros((len(readings), ))

    for i in range(len(readings)):
        if assignments is not None:
            labels[i] = assignments[i]
        for j in range(i+1, len(readings)):
            if time_series:
                dist = clustering.metrics.dtw(readings[i], readings[j])
            else:
                dist = clustering.metrics.euclidean(readings[i], readings[j])

            distances[i][j] = dist
            distances[j][i] = dist

    return distances, labels


def reverse_assignments(assignments):
    new_assignments = dict()

    for key in assignments.keys():
        indexes_list = assignments[key]
        for i in indexes_list:
            new_assignments[i] = key

    return new_assignments


def is_weekday(day_date):
    """
    Receives a given date as a datetime.datetime type, and returns a boolean variable, signalling whether or not the
    date in question is a weekday.
    :param day_date: The date to be processed, as a datetime.datetime type
    :return: A boolean variable, signalling whether or not the date in question is a weekday.
    """
    if not isinstance(day_date, datetime.datetime):
        return TypeError('Argument <day_date> should be of type <datetime.datetime>!')

    # datetime.date.weekday() - "Return day of the week, where Monday == 0 ... Sunday == 6."
    # So it is weekend if it returns 5 or 6, a lower value indicates weekday
    return day_date.weekday() < 5


