# coding: utf-8

import numpy

__author__ = 'Joaquim Leitão'
__copyright__ = 'Copyright (c) 2017 Joaquim Leitão'
__email__ = 'jocaleitao93@gmail.com'


def main():
    with open('arima/output.txt', 'r') as arima_output:
        arima_contents = arima_output.readlines()

    with open('polyfit/output.txt', 'r') as polyfit_output:
        polyfit_contents = polyfit_output.readlines()

    with open('kalman/output.txt', 'r') as kalman_output:
        kalman_contents = kalman_output.readlines()

    arima_rmses = [float(line[line.find('RMSE: ') + len('RMSE: '):]) for line in arima_contents]
    polyfit_rmses = [float(line[line.find('RMSE: ') + len('RMSE: '):]) for line in polyfit_contents]
    kalman_rmses = [float(line[line.find('RMSE: ') + len('RMSE: '):]) for line in kalman_contents]

    # Compare average rmse
    arima_mean_rmse = numpy.mean(arima_rmses)
    polyfit_mean_rmse = numpy.mean(polyfit_rmses)
    kalman_mean_rmse = numpy.mean(kalman_rmses)

    min_value = min(arima_mean_rmse, polyfit_mean_rmse, kalman_mean_rmse)

    if arima_mean_rmse == min_value:
        print('ARIMA produced better average results with full data (ARIMA:' + str(arima_mean_rmse) + '; Polyfit: ' +
              str(polyfit_mean_rmse) + '; Kalman: ' + str(kalman_mean_rmse) + ')')
    elif polyfit_mean_rmse == min_value:
        print('Polyfit produced better average results with full data (ARIMA:' + str(arima_mean_rmse) + '; Polyfit: ' +
              str(polyfit_mean_rmse) + '; Kalman: ' + str(kalman_mean_rmse) + ')')
    elif kalman_mean_rmse == min_value:
        print('Kalman produced better average results with full data (ARIMA:' + str(arima_mean_rmse) + '; Polyfit: ' +
              str(polyfit_mean_rmse) + '; Kalman: ' + str(kalman_mean_rmse) + ')')

    # Compare number of cases one approach was better than the other
    arima_better = 0
    polyfit_better = 0
    kalman_better = 0
    for i in range(len(arima_rmses)):
        min_value = min(arima_rmses[i], polyfit_rmses[i], kalman_rmses[i])
        if arima_rmses[i] == min_value:
            arima_better += 1
        elif polyfit_rmses[i] == min_value:
            polyfit_better += 1
        elif kalman_rmses[i] == min_value:
            kalman_better += 1

    print('ARIMA:' + str(arima_better) + '; Polyfit: ' + str(polyfit_better) + '; Kalman: ' + str(kalman_better))

if __name__ == '__main__':
    main()
