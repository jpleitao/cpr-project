# coding: utf-8

"""
Script that contains all the K-Means Clustering-related code
"""

import os
import random
import pickle
import numpy
import enum

import clustering.dba
import clustering.metrics
import clustering.utils

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


class CentroidType(enum.Enum):
    """
    Simple class to represent an enum for the centroid computation method
    """
    AVERAGE = 1
    DBA = 2
    MEDOID = 3


def _converged(centroids, centroids_old):
    if centroids_old is None:
        return False

    result = numpy.in1d(centroids, centroids_old)

    for temp in result:
        if not temp:
            return False
    return True


def k_means(readings, num_clusters, time_series=None, compute_centroid=None):
    if time_series is None:
        time_series = True
    elif not isinstance(time_series, bool):
        raise TypeError('Argument <time_series> must be of type <bool>!')

    if compute_centroid is None:
        compute_centroid = CentroidType.AVERAGE
    elif not isinstance(compute_centroid, CentroidType):
        raise TypeError('Argument <compute_centroid> must be of type <CentroidType.class>!')

    # Choose random centroids
    centroids_index = random.sample(range(len(readings)), num_clusters)
    centroids = readings[centroids_index]
    centroids_old = None

    iteration = 1
    assignments = dict()

    if compute_centroid == CentroidType.DBA:
        dba = clustering.dba.DBA(max_iter=30)
    else:
        dba = None

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
        if compute_centroid == CentroidType.DBA:
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
        elif compute_centroid == CentroidType.MEDOID:
            # Compute the centroid using the Medoid method
            # TODO: Test this!!!
            distances, _ = clustering.utils.compute_distances(time_series, readings, None)

            for key in assignments:

                min_index = None
                min_avg = float('inf')
                cur_elements = assignments[key]

                for my_i in range(len(cur_elements)):
                    avg = 0
                    for my_j in range(len(cur_elements)):
                        if my_j != my_i:
                            avg += distances[my_i][my_j]
                    avg = avg / (len(distances) - 1)

                    if avg < min_avg:
                        min_avg = avg
                        min_index = my_i

                centroids[key] = readings[min_index]

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


def save_best_results(best_results_list, time_series=None, compute_centroid=None, file_path=None):
    if time_series is None:
        time_series = True
    elif not isinstance(time_series, bool):
        raise TypeError('Argument <time_series> must be of type <bool>!')

    if compute_centroid is None:
        compute_centroid = CentroidType.AVERAGE
    elif not isinstance(compute_centroid, CentroidType):
        raise TypeError('Argument <compute_centroid> must be of type <CentroidType.class>!')

    if file_path is None:
        if time_series:
            if compute_centroid == CentroidType.DBA:
                file_path = os.getcwd() + '/data/best_results_dtw_dba.pkl'
            elif compute_centroid == CentroidType.AVERAGE:
                file_path = os.getcwd() + '/data/best_results_dtw.pkl'
            elif compute_centroid == compute_centroid.MEDOID:
                file_path = os.getcwd() + '/data/best_results_medoid.pkl'
        else:
            file_path = os.getcwd() + '/data/best_results_euclidean.pkl'
    with open(file_path, 'wb') as f:
        pickle.dump(best_results_list, f)


def load_best_results(time_series=None, compute_centroid=None, file_path=None):
    if time_series is None:
        time_series = True
    elif not isinstance(time_series, bool):
        raise TypeError('Argument <time_series> must be of type <bool>!')

    if compute_centroid is None:
        compute_centroid = CentroidType.AVERAGE
    elif not isinstance(compute_centroid, CentroidType):
        raise TypeError('Argument <compute_centroid> must be of type <CentroidType.class>!')

    if file_path is None:
        if time_series:
            if compute_centroid == CentroidType.DBA:
                file_path = os.getcwd() + '/data/best_results_dtw_dba.pkl'
            elif compute_centroid == CentroidType.AVERAGE:
                file_path = os.getcwd() + '/data/best_results_dtw.pkl'
            elif compute_centroid == compute_centroid.MEDOID:
                file_path = os.getcwd() + '/data/best_results_medoid.pkl'
        else:
            file_path = os.getcwd() + '/data/best_results_euclidean.pkl'
    try:
        with open(file_path, 'rb') as f:
            best_results = pickle.load(f)
    except Exception:
        print('Loading empty list')
        best_results = list()
    return best_results


def tune_kmeans(readings, k_values, number_runs, time_series=None, compute_centroid=None):
    if time_series is None:
        time_series = True
    elif not isinstance(time_series, bool):
        raise TypeError('Argument <time_series> must be of type <bool>!')

    if compute_centroid is None:
        compute_centroid = CentroidType.AVERAGE
    elif not isinstance(compute_centroid, CentroidType):
        raise TypeError('Argument <compute_centroid> must be of type <CentroidType.class>!')

    best_results = load_best_results(time_series, compute_centroid)

    for k in k_values:
        best_sc = -1
        best_assignments = None

        print('Running for k=' + str(k))

        for run in range(number_runs):
            centroids, assignments = k_means(readings, k, time_series, compute_centroid)
            # Process assignments from k_means to get the inverse (values -> keys instead of keys -> values)
            assignments_inverse = clustering.utils.reverse_assignments(assignments)
            # Run silhouette coefficient to evaluate centroids
            silhouette_coef = clustering.metrics.silhouette_coefficient(assignments_inverse, readings)

            if silhouette_coef > best_sc:
                best_sc = silhouette_coef
                best_assignments = assignments
            print('Run ' + str(run) + ' SC = ' + str(silhouette_coef) + ' ; Best_SC = ' + str(best_sc))

        best_results.append(_KMeansResults(k, best_assignments, best_sc))

    # Save best results to pickle file
    save_best_results(best_results, time_series=time_series, compute_centroid=compute_centroid)

    # FIXME: Just some debug print
    for current_result in best_results:
        print('K= ' + str(current_result.k) + ' and SC = ' + str(current_result.silhouette_coefficient))

    return best_results
