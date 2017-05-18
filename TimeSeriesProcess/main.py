# coding: utf-8

import os
import numpy

import preprocess
import clustering

__author__ = 'Joaquim Leitão'
__copyright__ = 'Copyright (c) 2017 Joaquim Leitão'
__email__ = 'jocaleitao93@gmail.com'


def do_preprocess(source_filepath):
    time, readings = preprocess.load_dataset(source_filepath)
    return preprocess.preprocess_run(time, readings)


def main():
    # readings_znormalise, data_transform = do_preprocess(os.getcwd() + '/data/data_final.csv')

    # FIXME: Testing-purposed Code -- Start
    # Load data from file
    imputed_file_path = os.getcwd() + '/data/imputed_data.csv'
    time, readings = preprocess.load_dataset(imputed_file_path, True)
    readings_znormalise = preprocess.z_normalise(readings)
    readings_index = preprocess.add_index_to_data(readings_znormalise)  # FIXME: Needed?

    # Load transformed data from file
    data_transform = numpy.genfromtxt(os.getcwd() + '/data/data_transformed_preprocessed.csv', delimiter=';',
                                      dtype=float)
    # FIXME: Testing-purposed Code -- End

    # Perform clustering with the standardised data and with the reduced data
    clustering.clustering_run(readings_znormalise, data_transform)

if __name__ == '__main__':
    main()
