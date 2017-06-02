# coding: utf-8

import os

import preprocess
import clustering

__author__ = 'Joaquim Leitão'
__copyright__ = 'Copyright (c) 2017 Joaquim Leitão'
__email__ = 'jocaleitao93@gmail.com'


def do_preprocess(source_filepath):
    time, readings = preprocess.load_dataset(source_filepath)
    return preprocess.preprocess_run(time, readings)


def main():
    readings_znormalise, data_transform, pca, scaler, means, stds = do_preprocess(os.getcwd() + '/data/data_final.csv')

    # Perform clustering with the standardised data and with the reduced data
    clustering.clustering_run(readings_znormalise, data_transform, pca, means, stds, scaler)

if __name__ == '__main__':
    main()
