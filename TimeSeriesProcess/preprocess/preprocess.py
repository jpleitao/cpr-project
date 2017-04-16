# coding: utf-8

import numpy
import matplotlib.pyplot as plt

__author__ = 'Joaquim Leitão'
__copyright__ = 'Copyright (c) 2017 Joaquim Leitão'
__email__ = 'jocaleitao93@gmail.com'


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


def get_unit_analysis(time, signal, non_nan_indexes, fs, plot=False):
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

    if plot or True:
        plt.figure()
        plt.plot(plot_time, readings)
        plt.title('Water consumption readings')
        plt.show()

        import sys
        sys.exit(-1)

    N = len(readings)

    # Compute the FFT and get the sample frequencies from the signal
    signal_fft = numpy.fft.fft(readings, N)
    freqs = numpy.fft.fftfreq(N, 1 / fs)
    # print(freqs.min(), freqs.max())

    if plot:
        plt.figure()
        plt.plot(abs(signal_fft))
        plt.title('Frequencies of the signal: abs(fft(signal))')
        plt.show()

    # Normally we would compute the maximum absolute value of the signal's FFT and get that frequency; however such
    # frequency corresponds to f=0 Hz. Therefore we will get the frequency of the second highest absolute value of the
    # signal's FFT
    # idx = numpy.argmax(numpy.abs(signal_fft))
    # freq_max_hz = abs(freqs[idx])
    abs_signal_fft = numpy.abs(signal_fft)
    idx = numpy.where(abs_signal_fft == second_largest(abs_signal_fft))
    idx = idx[0][0]

    return abs(freqs[idx])


def preprocess(time, readings):
    """
    Pre-processes the collected data:
        * Computes unit of analysis based on non-NaN values
    :param time: The time instants when data was collected
    :param readings: Collected data values
    """
    # Check parameters type
    if not isinstance(time, numpy.ndarray) or not isinstance(readings, numpy.ndarray):
        raise TypeError('Input parameters must be of type <numpy.array>!!')

    # Get nan and non-nan data
    nan_indexes = numpy.where(readings == 'n/a')
    non_nan_indexes = numpy.where(readings != 'n/a')
    # nan_readings = readings[nan_indexes]
    # nan_time = time[nan_indexes]

    fs = 1/60  # Data collected every minute, therefore Ts = 60 seconds

    # ================================== Compute unit of analysis ======================================================
    unit_analysis = get_unit_analysis(time, readings, non_nan_indexes, fs)
    unit_analysis_hours = (1/unit_analysis)/3600
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

    # =========================================== Fill missing values ==================================================
    # To fill the missing values we are going to take advantage of our unit of analysis: We are going to fit a linear
    # regression to each unit of analysis where there are missing values - This seems to make more sense then trying to
    # fit a linear regression on the entire dataset
    # TODO: Falar braz sobre isto, perguntar opinião (Fit janela vs todo o dataset)
    # TODO: Experimentar diferentes graus dos polinómios com base no número de concavidades das curvas; fazer alguns
    #       testes com dados completos - Isto e, pegar num dia que nao tenha NaN e "à mão" colocar alguns, fazer fit de
    #       um modelo e usá-lo para estimar os valores em falta. Comparar com os valores reais e calcular o RMSE

    # ====================================== Merge data in the unit of analysis ========================================
    # TODO: Merge the collected data into one-day vectors composed by 24 readings (sum the values in m3/h)!

    # ====================================== Save the data in an Excel file ============================================
