# coding: utf-8

import os

import preprocess

__author__ = 'Joaquim Leitão'
__copyright__ = 'Copyright (c) 2017 Joaquim Leitão'
__email__ = 'jocaleitao93@gmail.com'


def do_preprocess(source_filepath):
    time, readings = preprocess.load_dataset(source_filepath)
    preprocess.preprocess_run(time, readings)


def main():
    original_file_path = os.getcwd() + '/data/data_final.csv'
    do_preprocess(original_file_path)

if __name__ == '__main__':
    main()
