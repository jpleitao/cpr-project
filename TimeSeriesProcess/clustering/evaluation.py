# coding: utf-8

import os
import matplotlib.pyplot as plt
import numpy
import datetime

import clustering.k_means
import clustering.metrics
import clustering.utils

__author__ = 'Joaquim Leitão'
__copyright__ = 'Copyright (c) 2017 Joaquim Leitão'
__email__ = 'jocaleitao93@gmail.com'


def reconstruct_znormalisation(readings, means, stds):
    return (readings * stds) + means


def reconstruct_minmax_normalisation(readings, pca, scaler):
    data_reconstructed = pca.inverse_transform(readings)
    return scaler.inverse_transform(data_reconstructed)


def plot_centroids(centroids, clusters_image_path, k, time_series, pca, means, stds, scaler):
    folder_path = clusters_image_path + str(k) + '/'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    fig = plt.figure()
    centrois_list = list()

    # Plot the centroids
    for i in range(len(centroids)):
        centroid = centroids[i]

        # Reconstruct centroid to original time series
        if time_series:
            # Reverse the z-normalisation
            time = [j for j in range(0, 24)]
            centroid = reconstruct_znormalisation(centroid, means, stds)
        else:
            # Reverse the PCA and then reverse the min-max normalisation
            centroid = reconstruct_minmax_normalisation(centroid, pca, scaler)
            time = [j for j in range(len(centroid))]

        centrois_list.append(centroid)

        plt.plot(time, centroid, label='Cluster ' + str(i))
    plt.legend()
    plt.title('Cluster Centroids')
    fig.savefig(folder_path + 'centroids_' + str(k))
    plt.close(fig)

    return numpy.array(centrois_list)


def count_days(assignments):
    first_day = datetime.datetime(2016, 1, 1)

    weekdays = 0
    weekends = 0

    for assign in assignments:
        curr_delta = datetime.timedelta(days=assign)
        curr_date = first_day + curr_delta

        if clustering.utils.is_weekday(curr_date):
            weekdays += 1
        else:
            weekends += 1

    return weekdays, weekends


def count_months(assignments):
    first_day = datetime.datetime(2016, 1, 1)
    months = [0 for _ in range(12)]

    for assign in assignments:
        curr_delta = datetime.timedelta(days=assign)
        curr_date = first_day + curr_delta

        months[curr_date.month - 1] += 1

    return months


def further_cluster_evaluation(data, time_series, centroid_type, k_values, pca, means, stds, scaler):
    if not isinstance(time_series, bool):
        raise TypeError('Argument <time_series> must be of type <bool>!')

    if not isinstance(centroid_type, clustering.k_means.CentroidType):
        raise TypeError('Argument <centroid_type> must be of type <clustering.k_means.CentroidType>!')

    clusters_image_path = 'data/images/clustering/'
    if time_series:
        clusters_image_path += 'raw_'
    else:
        clusters_image_path += 'reduced_'

    if centroid_type == clustering.k_means.CentroidType.AVERAGE:
        clusters_image_path += 'average'
    elif centroid_type == clustering.k_means.CentroidType.DBA:
        clusters_image_path += 'dba'

    # Get best centroids
    best_results = clustering.k_means.load_best_results(time_series, centroid_type)

    centroids_dict = dict()
    assignments_dict = dict()

    for result in best_results:
        if result.k in k_values:
            centroids = plot_centroids(result.centroids, clusters_image_path, result.k, time_series, pca, means, stds,
                                       scaler)

            centroids_dict[result.k] = centroids
            assignments_dict[result.k] = result.assignments

    # Plot each centroid in an individual plot with the variation of the hourly errors: for each hour compute the
    # average dtw of the centroid to all the other elements
    for k in k_values:
        centroids_list = centroids_dict[k]
        folder_path = clusters_image_path + str(k) + '/'

        current_assignments = assignments_dict[k]

        for i in range(len(centroids_list)):
            # Get the readings assigned to this centroid
            centroid = centroids_list[i]
            curr_centroid_readings = data[current_assignments[i]]

            # Compute average dtw on each hour between the centroid and the readings
            average_dists = compute_average_hourly_dtw(centroid, curr_centroid_readings, time_series, pca, scaler,
                                                       means, stds)

            fig = plt.figure()
            time = [j for j in range(len(centroid))]
            plt.errorbar(time, centroid, yerr=average_dists, ecolor='g')
            plt.title('Cluster ' + str(i))
            fig.savefig(folder_path + 'centroid_' + str(i))
            plt.close(fig)

    # Compute percentages:
    #    * Percentage of days of the entire dataset - Title of the Pie Charts
    #    * Percentage of weekdays and weekends in each cluster - Pie Chart
    #    * Percentage of days of each month in each cluster - Pie chart
    for k in k_values:
        folder_path = clusters_image_path + str(k) + '/'
        current_assignments = assignments_dict[k]

        print('********************* k = ' + str(k) + ' *********************')

        for cluster in current_assignments:
            assignments_cluster = current_assignments[cluster]
            percentage_dataset_cluster = (len(assignments_cluster) / 365) * 100

            print('[' + str(time_series) + '] % Days = ' + str(percentage_dataset_cluster) + ' Total= ' +
                  str(len(assignments_cluster)) + ' For cluster ' + str(cluster))

            weekdays, weekends = count_days(assignments_cluster)

            # Weekdays vs weekends
            labels = ['Weekdays', 'Weekends']
            sizes = [weekdays, weekends]
            colours = ['lightskyblue', 'yellowgreen']

            fig = plt.figure()
            plt.pie(x=sizes, labels=labels, explode=(0.1, 0), colors=colours, shadow=True, autopct='%1.2f%%')
            plt.axis('equal')
            plt.title('Cluster ' + str(cluster) + '; Percentage of days=' + str(round(percentage_dataset_cluster, 2)) +
                      '%')
            fig.savefig(folder_path + 'percent_dataset' + str(cluster))
            plt.close(fig)

            # Days of each month - Pie chart with 12 "cases"
            labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'June', 'July', 'Aug', 'Sept', 'Oct', 'Nov', 'Dec']
            sizes = count_months(assignments_cluster)
            explode = (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)

            fig = plt.figure()
            plt.pie(x=sizes, labels=labels, explode=explode, shadow=True, autopct='%1.2f%%')
            plt.axis('equal')
            plt.title('Cluster ' + str(cluster) + '; Percentage of days=' + str(round(percentage_dataset_cluster, 2)) +
                      '%')
            fig.savefig(folder_path + 'percent_months' + str(cluster))
            plt.close(fig)


def compute_average_hourly_dtw(centroid, centroid_readings, time_series, pca, scaler, means, stds):
    ncols = centroid.shape[0]
    average_dists = [0 for _ in range(ncols)]

    for i in range(len(centroid_readings)):
        curr_reading = centroid_readings[i]

        # Reconstruct current reading
        if time_series:
            # Reverse the z-normalisation
            curr_reading = reconstruct_znormalisation(curr_reading, means, stds)
        else:
            # Reverse the PCA and then reverse the min-max normalisation
            curr_reading = reconstruct_minmax_normalisation(curr_reading, pca, scaler)

        for j in range(ncols):
            curr_dist = clustering.metrics.euclidean(centroid[j], curr_reading[j])
            average_dists[j] += curr_dist
    for j in range(ncols):
        average_dists[j] = average_dists[j] / len(centroid_readings)
    return numpy.array(average_dists)
