# coding: utf-8

"""
Script that contains all the K-Means Clustering-related code
"""

import os
import random
import pickle
import numpy

import clustering.metrics
import clustering.dba

__author__ = 'Joaquim Leitão'
__copyright__ = 'Copyright (c) 2017 Joaquim Leitão'
__email__ = 'jocaleitao93@gmail.com'


class _KMeansResults(object):
    """
    Private class to store best cluster solutions using the K-Means clustering algorithm for a given value of K
    (A list of instances of this class is intended to be saved)
    """

    def __init__(self, k, assigments, silhouette_coefficient):
        self._k = k
        self._assignments = assigments
        self._sc = silhouette_coefficient

    @property
    def k(self):
        return self._k

    @property
    def assignments(self):
        return self._assignments

    @property
    def silhouette_coefficient(self):
        return self._sc


def _converged(centroids, centroids_old):
    if centroids_old is None:
        return False

    result = numpy.in1d(centroids, centroids_old)

    for temp in result:
        if not temp:
            return False
    return True


def k_means(readings, num_clusters, time_series=None, do_dba=None):
    if time_series is None:
        time_series = True
    elif not isinstance(time_series, bool):
        raise TypeError('Argument <time_series> must be of type <bool>!')

    if do_dba is None:
        do_dba = False
    elif not isinstance(do_dba, bool):
        raise TypeError('Argument <do_dba> must be of type <bool>!')

    # Choose random centroids
    centroids_index = random.sample(range(len(readings)), num_clusters)
    centroids = readings[centroids_index]
    centroids_old = None

    iteration = 1
    assignments = dict()

    if do_dba:
        dba = clustering.dba.DBA(max_iter=30)

    while not _converged(centroids, centroids_old) and iteration < 300:
        # Assign data points to clusters
        assignments = dict()

        for ind, i in enumerate(readings):
            min_dist = float('inf')
            closest_clust = None

            # Compute closest centroid to data point
            for c_ind, j in enumerate(centroids):
                if time_series:
                    if clustering.metrics.lb_keogh(i, j, 5) < min_dist:
                        cur_dist = clustering.metrics.dtw(i, j)
                    else:
                        cur_dist = float('inf')
                else:
                    cur_dist = clustering.metrics.euclidean(i, j)

                if cur_dist < min_dist:
                    min_dist = cur_dist
                    closest_clust = c_ind

            if closest_clust in assignments:
                assignments[closest_clust].append(ind)
            else:
                assignments[closest_clust] = [ind]

        # Backup centroids
        centroids_old = numpy.copy(centroids)

        # Recalculate centroids of clusters
        if do_dba:
            # Compute the centroid using the Dynamic Time Warping Barycenter Average (DBA) method
            for key in assignments:
                cluster_time_series_list = list()
                for k in assignments[key]:
                    cluster_time_series_list.append(readings[k])
                cluster_time_series_list = numpy.array(cluster_time_series_list)
                len_series = len(cluster_time_series_list[0])

                cluster_time_series_list = clustering.dba.ts_to_dba_list(cluster_time_series_list)
                result = dba.compute_average(cluster_time_series_list, dba_length=len_series)

                centroids[key] = result.reshape((1, len(result)))[0]
        else:
            # Compute centroid as average of all time series in the cluster
            for key in assignments:
                clust_sum = 0
                for k in assignments[key]:
                    clust_sum = clust_sum + readings[k]
                centroids[key] = [m / len(assignments[key]) for m in clust_sum]

        iteration += 1

    print('Converged in iteration ' + str(iteration))
    return centroids, assignments


def save_best_results(best_results_list, time_series=None, do_dba=None, file_path=None):
    if time_series is None:
        time_series = True
    elif not isinstance(time_series, bool):
        raise TypeError('Argument <time_series> must be of type <bool>!')

    if do_dba is None:
        do_dba = False
    elif not isinstance(do_dba, bool):
        raise TypeError('Argument <do_dba> must be of type <bool>!')

    if file_path is None:
        if time_series:
            if do_dba:
                file_path = os.getcwd() + '/data/best_results_dtw_dba.pkl'
            else:
                file_path = os.getcwd() + '/data/best_results_dtw.pkl'
        else:
            file_path = os.getcwd() + '/data/best_results_euclidean.pkl'
    with open(file_path, 'wb') as f:
        pickle.dump(best_results_list, f)


def load_best_results(time_series=None, do_dba=None, file_path=None):
    if time_series is None:
        time_series = True
    elif not isinstance(time_series, bool):
        raise TypeError('Argument <time_series> must be of type <bool>!')

    if do_dba is None:
        do_dba = False
    elif not isinstance(do_dba, bool):
        raise TypeError('Argument <do_dba> must be of type <bool>!')

    if file_path is None:
        if time_series:
            if do_dba:
                file_path = os.getcwd() + '/data/best_results_dtw_dba.pkl'
            else:
                file_path = os.getcwd() + '/data/best_results_dtw.pkl'
        else:
            file_path = os.getcwd() + '/data/best_results_euclidean.pkl'
    try:
        with open(file_path, 'rb') as f:
            best_results = pickle.load(f)
    except Exception:
        print('Loading empty list')
        best_results = list()
    return best_results


def reverse_assignments(assignments):
    new_assignments = dict()

    for key in assignments.keys():
        indexes_list = assignments[key]
        for i in indexes_list:
            new_assignments[i] = key

    return new_assignments


def tune_kmeans(readings, k_raw_data, number_runs, time_series=None, do_dba=None):
    if time_series is None:
        time_series = True
    elif not isinstance(time_series, bool):
        raise TypeError('Argument <time_series> must be of type <bool>!')

    if do_dba is None:
        do_dba = False
    elif not isinstance(do_dba, bool):
        raise TypeError('Argument <do_dba> must be of type <bool>!')

    best_results = load_best_results(time_series, do_dba)

    for k in k_raw_data:
        best_sc = -1
        best_assignments = None

        print('Running for k=' + str(k))

        for run in range(number_runs):
            centroids, assignments = k_means(readings, k, time_series, do_dba)
            # Process assignments from k_means to get the inverse (values -> keys instead of keys -> values)
            assignments_inverse = reverse_assignments(assignments)
            # Run silhouette coefficient to evaluate centroids
            silhouette_coef = clustering.metrics.silhouette_coefficient(assignments_inverse, readings)

            if silhouette_coef > best_sc:
                best_sc = silhouette_coef
                best_assignments = assignments
            print('Run ' + str(run) + ' SC = ' + str(silhouette_coef) + ' ; Best_SC = ' + str(best_sc))

        best_results.append(_KMeansResults(k, best_assignments, best_sc))

    # Save best results to pickle file
    save_best_results(best_results, time_series=time_series, do_dba=do_dba)

    # FIXME: Just some debug print
    for current_result in best_results:
        print('K= ' + str(current_result.k) + ' and SC = ' + str(current_result.silhouette_coefficient))

    return best_results
