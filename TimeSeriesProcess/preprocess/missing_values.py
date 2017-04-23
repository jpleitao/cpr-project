# coding: utf-8

import random

import matplotlib.pyplot as plt
import numpy
import numpy.polynomial.polynomial

__author__ = 'Joaquim Leitão'
__copyright__ = 'Copyright (c) 2017 Joaquim Leitão'
__email__ = 'jocaleitao93@gmail.com'


def _fit_polyfit(time, readings, degree=3):
    coefs = numpy.polynomial.polynomial.polyfit(time, readings, degree)
    return numpy.polynomial.polynomial.Polynomial(coefs)


def _predict_polyfit(model, pos):
    return model(pos)


def test_imputation(current_readings, day):
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

    time = list()
    values = list()

    for i in range(len(day_readings)):
        if i not in positions:
            time.append(i)
            values.append(day_readings[i])

    time = numpy.array(time)
    values = numpy.array(values)

    model = _fit_polyfit(time, values, 8)

    rmse = 0
    for pos in positions:
        estimate = _predict_polyfit(model, pos)

        real_value = day_readings[pos]
        # print('[Position=' + str(pos) + '] Got ' + str(estimate) + ' and real value is ' + str(real_value))
        rmse += (real_value - estimate)**2

    rmse = numpy.sqrt(1/len(positions) * rmse)

    print('[Test_Imputation]DAY: ' + str(day) + ' positions: ' + str(positions) + ' RMSE: ' + str(rmse))

    fig = plt.figure()
    time = [i for i in range(0, 24)]
    estimates = [_predict_polyfit(model, i) for i in time]

    plt.plot(time, day_readings, 'g', label='Real Values')
    plt.plot(time, estimates, 'k', label='Estimates')
    plt.legend()
    plt.title('Day ' + str(day))
    fig.savefig('/data/missing_data/polyfit/day_' + str(day))
    plt.close(fig)


def test_polyfit_missing(merged_data_filepath):
    # Read values from csv into numpy array
    values = numpy.genfromtxt(merged_data_filepath, delimiter=';', dtype=str)
    rows, columns = values.shape

    # Process each unit of analysis - If a nan is found then fit a model to that day and impute missing values
    for i in range(rows):
        have = False
        for j in range(1, columns):
            if values[i][j] == 'nan':
                have = True
        if not have:
            # Impute
            test_imputation(values[i], i)
