# coding: utf-8

import numpy
from datetime import datetime

__author__ = 'Joaquim Leitão'
__copyright__ = 'Copyright (c) 2017 Joaquim Leitão'
__email__ = 'jocaleitao93@gmail.com'


def load_dataset(source_filepath):
    values = numpy.genfromtxt(source_filepath, delimiter=';', dtype=str)

    time = values[:, 0]
    time = list(map(lambda s: datetime.strptime(s, '%d/%m/%Y %H:%M:%S'), time))
    time = numpy.array(time)
    readings = values[:, 1]
    # Replace ',' with '.' for floating-point
    readings = list(map(lambda s: s.replace(',', '.'), readings))
    readings = numpy.array(readings)

    return time, readings
