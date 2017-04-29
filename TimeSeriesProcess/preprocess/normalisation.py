# coding: utf-8

import numpy
import sklearn.preprocessing

__author__ = 'Joaquim Leitão'
__copyright__ = 'Copyright (c) 2017 Joaquim Leitão'
__email__ = 'jocaleitao93@gmail.com'


def normalise_min_max(readings):
    scaler = sklearn.preprocessing.MinMaxScaler()
    scaler.fit(readings)
    return scaler.transform(readings)


def z_normalise(readings):
    return (readings - numpy.mean(readings, axis=0))/numpy.std(readings, axis=0)
