# coding: utf-8

"""
Contains utility functions used throughout the clustering package
"""

import numpy

import clustering.metrics

__author__ = 'Joaquim Leitão'
__copyright__ = 'Copyright (c) 2017 Joaquim Leitão'
__email__ = 'jocaleitao93@gmail.com'


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


