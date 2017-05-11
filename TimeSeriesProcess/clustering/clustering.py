# coding: utf-8

import scipy.cluster.hierarchy
import scipy.spatial.distance
import clustering.distances

import matplotlib.pyplot as plt

__author__ = 'Joaquim Leitão'
__copyright__ = 'Copyright (c) 2017 Joaquim Leitão'
__email__ = 'jocaleitao93@gmail.com'


def hierarchical(readings, time_series=None):
    if time_series is None:
        time_series = True
    elif not isinstance(time_series, bool):
        raise TypeError('Argument <time_series> must be of type <bool>!')

    if time_series:
        # Hierarchical clustering of raw time series data - Use Dynamic Time Warping
        distance_metric = clustering.distances.dtw
    else:
        # Hierarchical clustering of transformed time series data - Use Euclidean distance
        distance_metric = 'euclidean'

    z = scipy.cluster.hierarchy.linkage(y=readings, method='average', metric=distance_metric)
    plt.figure()
    scipy.cluster.hierarchy.dendrogram(z)


def k_means():
    pass


def clustering_run(readings, data_transform):
    """
    Performs clustering of the time series data. The same clustering methods are intended to be applied to two different
    time series representations: On one of them the raw time series data is intended to be used; on the other the PCA
    dimensionality reduction algorithm was applied to reduce the time series values to a dimensionality capable of
    explaining 80% of the data variance (3 dimensions)
    :param readings: The original time series values
    :param data_transform: The time series values in the reduced (that is, transformed) dimensionality
    """
    # ==================================== Hierarchical Clustering =====================================================
    # Start by performing hierarchical clustering on the original, raw, time series data. For such a data representation
    # an appropriate distance metric must be used, which in this case is the Dynamic Time Warping.
    # We then move on to performing the same task on the transformed data (reduced dimensionality)
    hierarchical(readings)
    hierarchical(data_transform, False)
    plt.show()

    # We will start the analysis of the results with the Hierarchical Clustering on the transformed data.
    # In this task time series data was transformed with PCA in order to reduce its dimensionality. A transformation
    # guided by the eigenvectors of the data that explain 80% of its variance was performed on the data prior to the
    # application of the hierarchical clustering algorithm. With respect to this algorithm, the average linkage method
    # and the euclidean distance metrics were used. A dendrogram was produced as the outcome of this task
    # Analysing the produced dendrogram 2 main clusters can be identified, the first around distance 100 and the second
    # around distance 75. As the distance gets shorter, more clusters can be identified. Falling between distance 75
    # and 50 two more merging points can be identified, and another right below distance 50. This exercise can be
    # extended to smaller distances, so we can obtain up to 8 clusters. If we continue the exercise beyond this point
    # the number of clusters rapidly increases. As a result, from the visual inspection of the dendrogram a number of
    # clusters between 4 and 8 seems to be a good guess, and therefore is intended to be explored in the initial
    # experiments carried out with the K-Means Clustering algorithm.
    #
    # Moving on to the Hierarchical Clustering on the raw time series data, a different scenario is observed. In this
    # exercise, as stated, raw time series data was clustered using the Dynamic Time Warping (DTW) distance metric. As
    # our samples contain 24 elements, instead of the 3 directions chosen by the PCA in the previous exercise, the
    # distances presented in the dendrogram are considerably higher in this exercise.
    # Again, two main clusters are identified. Unlike in the previous exercise, the left side of the dendrogram
    # (represented in green) will only be further divided for significantly smaller distances, for which a high number
    # of clusters needs to be considered. As a result, this will remain as one cluster. Analysing the right side of the
    # dendrogram (represented in red), after the division around distance 16000 the number of clusters gets too high (a
    # lot of divisions for small decreases in the distance). As a result, the number of clusters in this distance will
    # be considered as a reference in the initial exploratory analysis. Such a number is 4.
    # In this sense, a number of clusters between 2 and 4 are intended to be explored in the K-Means Clustering
    # algorithm. If the cluster evaluation metrics for these limits suggest an invalid cluster formation, this range may
    # be extended.
