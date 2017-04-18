# coding: utf-8

import csv
import random

import numpy
import numpy.polynomial.polynomial
import datetime
import matplotlib.pyplot as plt

from statsmodels.tsa.arima_model import ARIMA

__author__ = 'Joaquim Leitão'
__copyright__ = 'Copyright (c) 2017 Joaquim Leitão'
__email__ = 'jocaleitao93@gmail.com'


def fit_polyfit(time, readings, degree=3):
    coefs = numpy.polynomial.polynomial.polyfit(time, readings, degree)
    return numpy.polynomial.polynomial.Polynomial(coefs)


def predict_polyfit(model, pos):
    return model(pos)


def fit_arima(time, readings):
    # FIXME: FIX THIS!
    # See: http://statsmodels.sourceforge.net/devel/examples/notebooks/generated/tsa_arma_1.html
    print(time)
    time_values = list()
    for sample in time:
        time_values.append(datetime.datetime(year=2016, month=1, day=1, hour=sample))

    values = numpy.column_stack((time_values, readings))

    model = ARIMA(values, order=(5, 1, 0))
    model_fit = model.fit(disp=True)

    print(model_fit.summary())


def impute_day(current_readings):
    # Get just the readings and convert to an array of floats
    day_readings = current_readings[1:].astype(numpy.float64)

    # FIXME: Test with ARIMA and Linear Regression
    # FIXME: Maybe separate the data into nan and non nan? Also make use of hour information?
    # FIXME: We already have the position of the nan data, so it would be easy to get the hour


def test_imputation(current_readings):
    # Get just the readings and convert to an array of floats
    day_readings = current_readings[1:].astype(numpy.float64)

    # Select number of missing data
    numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    num_missing = numbers[random.randint(0, len(numbers)-1)]

    possible_positions = [i for i in range(len(day_readings))]
    positions = []
    for j in range(num_missing):
        index = random.randint(0, len(possible_positions)-1)
        positions.append(possible_positions[index])
        possible_positions.remove(possible_positions[index])

    time = []
    values = []
    for i in range(len(day_readings)):
        if i not in positions:
            time.append(i)
            values.append(day_readings[i])

    time = numpy.array(time)
    values = numpy.array(values)

    # model = fit_polyfit(time, values, 8)
    model = fit_arima(time, values)

    return
    rmse = 0
    for pos in positions:
        # estimate = predict_polyfit(model, pos)
        estimate = predict_polynomial_features(model, pos)

        real_value = day_readings[pos]
        print('[Position=' + str(pos) + '] Got ' + str(estimate) + ' and real value is ' + str(real_value))
        rmse += (real_value - estimate)**2

    rmse = numpy.sqrt(1/len(positions) * rmse)
    print(rmse)

    plt.figure()
    time = [i for i in range(0, 24)]
    # estimates = [predict_polyfit(model, i) for i in time]
    estimates = [predict_polynomial_features(model, i) for i in time]

    plt.plot(time, day_readings, 'g')
    plt.plot(time, estimates, 'k')
    plt.draw()


def fill_missing_values(merged_data_filepath):
    # Read values from csv into numpy array
    values = numpy.genfromtxt(merged_data_filepath, delimiter=';', dtype=str)
    rows, columns = values.shape

    # Fazer varios plots dos consumos por dia e estimar ordem polinomio - Modelo de 2ª ou 3ª ordem
    # FIXME: Testar primeiro com dias completos - Colocar uma percentagem de valores em falta (uns salpicados, outros
    # seguidos) estimar modelo e ver RMSE

    # Process each unit of analysis - If a nan is found then fit a model to that day and impute missing values
    for i in range(rows):
        have = False
        for j in range(1, columns):
            if values[i][j] == 'nan':
                have = True
                """
                impute_day(values[i])

                # FIXME: Just for testing we do it in the first day and stop
                return
                # break
                """
        if not have:
            # Impute
            test_imputation(values[i])
            return

    plt.show()
