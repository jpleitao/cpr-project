# coding: utf-8

import os

import csv
from datetime import datetime

import numpy
import matplotlib.pyplot as plt

import preprocess

__author__ = 'Joaquim Leitão'
__copyright__ = 'Copyright (c) 2017 Joaquim Leitão'
__email__ = 'jocaleitao93@gmail.com'


def load_dataset_raw_readings(source_filepath):
    values = numpy.genfromtxt(source_filepath, delimiter=';', dtype=str)

    time = values[:, 0]
    time = list(map(lambda s: datetime.strptime(s, '%d/%m/%Y %H:%M'), time))
    time = numpy.array(time)
    readings = values[:, 1]
    # Replace ',' with '.' for floating-point
    readings = list(map(lambda s: s.replace(',', '.'), readings))
    readings = numpy.array(readings)

    return time, readings


def load_dataset_days(source_filepath):
    values = numpy.genfromtxt(source_filepath, delimiter=';', dtype=str)

    time = values[:, 0]
    time = list(map(lambda s: datetime.strptime(s, '%Y-%m-%d %H:%M:%S'), time))
    time = numpy.array(time)

    readings = values[:, 1:]
    readings = readings.astype(numpy.float)

    return time, readings


def load_dataset(source_filepath, aggregated=False):
    if aggregated:
        return load_dataset_days(source_filepath)
    else:
        # Sample csv file with two collumns
        return load_dataset_raw_readings(source_filepath)


def second_largest(numbers):
    count = 0
    m1 = m2 = float('-inf')
    for x in numbers:
        count += 1
        if x > m2:
            if x >= m1:
                m1, m2 = x, m1
            else:
                m2 = x
    return m2 if count >= 2 else None


def get_unit_analysis(time, signal, fs, plot=False):
    """
    Computes the unit of analysis for the given signal, which is given by the frequency with higher contribution to the
    signal, other than 0
    :param time: A numpy array with the time of the readings
    :param signal: A numpy array with the readings
    :param fs: The sampling frequency
    :param plot: A boolean variable, stating whether or not a plot with the readings over time is to be presented
    :return: The unit of analysis for the signal: the frequency - Other than 0 Hz - with highest contribution to the
            signal
    """
    non_nan_indexes = numpy.where(signal != 'n/a')
    non_nan_time = time[non_nan_indexes]
    non_nan_readings = signal[non_nan_indexes].astype(float)

    plot_time = list()
    readings = list()

    # This should be performed without missing values, otherwise I am not respecting the sampling rate. In
    # previous versions of the code this was performed during periods of time in which data was continuous (no missing
    # values) and similar results were obtained (exactly 24h were obtained in some cases). As a final test and
    # experiment it was decided to compute the unit of analysis for the entire non-NaN dataset
    for i in range(len(non_nan_time)):
        plot_time.append(non_nan_time[i])
        readings.append(non_nan_readings[i])

    if plot:
        plt.figure()
        plt.plot(plot_time, readings)
        plt.title('Water consumption readings')
        plt.show()

    n = len(readings)

    # Compute the FFT and get the sample frequencies from the signal
    signal_fft = numpy.fft.fft(readings, n)
    freqs = numpy.fft.fftfreq(n, 1 / fs)
    # print(freqs.min(), freqs.max())

    if plot:
        plt.figure()
        plt.plot(abs(signal_fft))
        plt.title('Frequencies of the signal: abs(fft(signal))')
        plt.show()

    # Normally we would compute the maximum absolute value of the signal's FFT and get that frequency; however such
    # frequency corresponds to f=0 Hz. Therefore we will get the frequency of the second highest absolute value of the
    # signal's FFT
    abs_signal_fft = numpy.abs(signal_fft)

    idx = numpy.argmax(abs_signal_fft)
    freq_max_hz = abs(freqs[idx])
    if freq_max_hz == 0:
        idx = numpy.where(abs_signal_fft == second_largest(abs_signal_fft))
        idx = idx[0][0]
        freq_max_hz = abs(freqs[idx])

    return freq_max_hz


def merge_data_readings(time, readings, unit_analysis_hours, step_minutes=60):
    # The unit of analysis tells us the smallest sequence that we are going to analyse (one day, in our case).
    # 'Step minutes' gives us the time interval between samples of a given sequence (That is, it allows us to compute
    # the number of features of our sequences)
    number_features_sequence = numpy.ceil(unit_analysis_hours)
    number_samples_day = int(number_features_sequence * step_minutes)

    # The final data representation will be an array of arrays (list of lists) where each 'inner' array contains the
    # date of the readings and the readings themselves
    result = list()
    i = 0

    while i < len(time):
        current_list = [time[i]]

        print('Moving to day ' + str(time[i]))

        # Each day covers the samples in the range (i, i+number_samples_day) and we need to sum those samples
        # 'step_minutes' at a time
        j = 0
        while j < number_features_sequence:
            curr_sum = 0
            start_index = i + (j * step_minutes)
            end_index = i + ((j + 1) * step_minutes)

            for k in range(start_index, end_index):
                if readings[k] == 'n/a':
                    curr_sum = numpy.nan
                    break
                else:
                    curr_sum += float(readings[k])
            current_list.append(curr_sum)
            j += 1

        # Move to next day
        i += number_samples_day

        result.append(current_list)

    # Return result so it can be saved to file
    return result


def save_merged_data_excel(merged_data, path=None):
    if path is None:
        path = os.getcwd() + '/data/merged_data.csv'

    with open(path, 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter=';', quotechar='', quoting=csv.QUOTE_NONE)
        writer.writerows(merged_data)

    return path


def preprocess_run(time, readings):
    """
    Pre-processes the collected data:
        * Computes unit of analysis based on non-NaN values
    :param time: The time instants when data was collected
    :param readings: Collected data values
    """
    # Check parameters type
    if not isinstance(time, numpy.ndarray) or not isinstance(readings, numpy.ndarray):
        raise TypeError('Input parameters must be of type <numpy.array>!!')

    fs = 1 / 60  # Data collected every minute, therefore Ts = 60 seconds

    # ================================== Compute unit of analysis ======================================================
    unit_analysis = get_unit_analysis(time, readings, fs)
    unit_analysis_hours = (1 / unit_analysis) / 3600
    print('The unit of analysis is of ' + str(unit_analysis_hours) + ' hours')

    # Note: The frequency with the largest contribution to the resulting signal is the frequency 0. This result may seem
    # strange but could actually make sense if we consider the following: By performing the FFT of the signal an
    # approximation of the fundamental frequency is usually obtained by determining the frequency with higher
    # contribution to the signal, that is, with the highest amplitude. By performing such operations on our signal we
    # can see that the frequency with highest amplitude is the frequency f=0 Hz. Such a result can be explained by the
    # fact that the collected signal is not a periodic signal, meaning that its fundamental period is infinite and
    # therefore its fundamental frequency is 0.
    # However, the frequency with the second highest contribution was around 24h, which suggests that this could be our
    # unit of analysis

    # ====================================== Merge data in the unit of analysis ========================================
    # Merge the collected data into one-day vectors composed by 24 readings (sum the values in m3/h)!
    # Whenever a missing value is present in that hour consider that value NaN!
    merged_data = merge_data_readings(time, readings, unit_analysis_hours)
    merged_data_filepath = save_merged_data_excel(merged_data)

    # =========================================== Fill missing values ==================================================
    # To fill the missing values we are going to take advantage of our unit of analysis: We are going to fit a linear
    # regression to each unit of analysis where there are missing values - This seems to make more sense then trying to
    # fit a linear regression on the entire dataset
    # Look at the different profiles for one day (data hour by hour) and try to find out what could be the right order
    # for the fit

    # We want to try and compare three different approaches: Fitting a polynomial function to our data; Fitting an ARIMA
    # model to our data; Interpolating missing data with a kalman filter
    # The first approach will be implemented in Python, in the function:
    #           preprocess.test_polyfit_missing (preprocess/missing_values.py)
    # The second approach will be implemented in R, in the function:
    #           testArimaMissing (preprocess/missing_values.R)
    # The third approach will be implemented in R, in the function:
    #           testKalmanFilterMissing (preprocess/missing_values.R)
    #
    # Both approaches were initially applied to the days where no missing values had been registered: The idea was to
    # 'artificially' introduce missing data in those days and compare the imputations of both approaches. This
    # comparison was performed with the RMSE metric.
    # Missing data were imputed based on the unit of analysis: That is, for each day missing data would be induced, the
    # corresponding models learned, estimation values generated and the corresponding RMSE values computed
    #
    # The comparison with the RMSE metric was performed in two stages:
    #  * Firstly, the average RMSE values of each approach were compared
    #  * Secondly, the total number of days were one approach registered a smaller RMSE than its alternatives were also
    #    compared
    fill_missing_values_test = False  # Not necessary in the final version, as this was implemented in R
    if fill_missing_values_test:
        merged_data_filepath = '/media/jpleitao/Data/PhD/PDCTI/CPR/cpr-project/TimeSeriesProcess/data/merged_data.csv'
        preprocess.polyfit_missing(merged_data_filepath)

    # The Polyfit approach registered inferior results in both stages when compared to the ARIMA approach:
    #   * An average RMSE of 12403.1671256 was registered against an average RMSE of 571.373059567 for ARIMA and
    #     263.833402729 for the Kalman filter
    #   * In only 4 of the days the polyfit approach registered a smaller RMSE than the ARIMA and the Kalman filter.
    #     By its turn, ARIMA registered smaller RMSEs in 72 days, and the Kalman filter was the best in 189 days
    #
    # Based on such results the Kalman Filter was chosen as the imputation approach for our data. Therefore, for each
    # day where missing data were registered, a Kalman Filter model was computed and estimations for the missing data
    # in that day were obtained. The implementation of this procedure can be found in the R function
    # 'fillMissingValuesKalman', implemented in the file located at 'preprocess/missing_values.R'

    # ========================================= Dimensionality Reduction ===============================================
    preprocessed_file_path = os.path.dirname(merged_data_filepath) + '/imputed_data.csv'
    # FIXME: Load dataset
    # preprocess.reduce_dimensionality()
