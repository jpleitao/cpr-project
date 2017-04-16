# coding: utf-8

from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model

__author__ = 'Joaquim Leitão'
__copyright__ = 'Copyright (c) 2017 Joaquim Leitão'
__email__ = 'jocaleitao93@gmail.com'


def fit_linear_regression(degree):
    poly = PolynomialFeatures(degree=degree)


def fit_arima():
    pass


def fill_missing_values(time, readings):
    # TODO: Fazer com linear regression e com ARIMA
    pass
