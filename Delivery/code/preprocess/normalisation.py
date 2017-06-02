# coding: utf-8

import numpy
import sklearn.preprocessing

__author__ = 'Joaquim Leitão'
__copyright__ = 'Copyright (c) 2017 Joaquim Leitão'
__email__ = 'jocaleitao93@gmail.com'


def normalise_min_max(readings, return_all=None):
    scaler = sklearn.preprocessing.MinMaxScaler()
    scaler.fit(readings)

    if return_all is None:
        return scaler.transform(readings)
    return scaler.transform(readings), scaler


def z_normalise(readings, return_all=None):
    means = numpy.mean(readings, axis=0)
    stds = numpy.std(readings, axis=0)

    if return_all is None:
        return (readings - numpy.mean(readings, axis=0))/numpy.std(readings, axis=0)
    return (readings - numpy.mean(readings, axis=0))/numpy.std(readings, axis=0), means, stds
