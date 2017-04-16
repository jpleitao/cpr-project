# coding: utf-8

import os

import preprocess

__author__ = 'Joaquim Leitão'
__copyright__ = 'Copyright (c) 2017 Joaquim Leitão'
__email__ = 'jocaleitao93@gmail.com'


def main(source_filepath):
    time, readings = preprocess.load_dataset(source_filepath)
    preprocess.preprocess(time, readings)

if __name__ == '__main__':
    _source_filepath = os.getcwd() + '/data/data_final.csv'
    main(_source_filepath)
