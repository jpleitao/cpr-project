# coding: utf-8

from preprocess.preprocess import load_dataset, preprocess_run, add_index_to_data
from preprocess.missing_values import polyfit_missing
from preprocess.normalisation import normalise_min_max, z_normalise
from preprocess.dimensionality_reduction import reduce_dimensionality

__author__ = 'Joaquim Leitão'
__copyright__ = 'Copyright (c) 2017 Joaquim Leitão'
__email__ = 'jocaleitao93@gmail.com'
