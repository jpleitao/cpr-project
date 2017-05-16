# coding: utf-8

import matplotlib.pyplot as plt

import clustering.metrics
import clustering.hierarchical
import clustering.k_means

__author__ = 'Joaquim Leitão'
__copyright__ = 'Copyright (c) 2017 Joaquim Leitão'
__email__ = 'jocaleitao93@gmail.com'


def clustering_run(readings, data_transform):
    """
    Performs clustering of the time series data. The same clustering methods are intended to be applied to two different
    time series representations: On one of them the raw time series data (with Z-normalisation) is intended to be used;
    on the other the PCA dimensionality reduction algorithm was applied to reduce the time series values to a
    dimensionality capable of explaining 80% of the data variance (3 dimensions)
    :param readings: The original time series values, with Z-Normalisation
    :param data_transform: The time series values in the reduced (that is, transformed) dimensionality
    """
    # ==================================== Hierarchical Clustering =====================================================
    # Start by performing hierarchical clustering on the raw, z-normalised, time series data. For such a data
    # representation an appropriate distance metric must be used, which in this case is the Dynamic Time Warping.
    # Mueen and Keogh highlight the importance of combining Z-Normalisation with DTW for Time Series Clustering in a
    # 2016 Conference Presentation:
    #   "Abdullah Mueen, Eamonn J. Keogh: Extracting Optimal Performance from Dynamic Time Warping. KDD 2016: 2129-2130"
    #
    # We then move on to performing the same task on the transformed data (reduced dimensionality)

    # FIXME: Uncomment this in the final version of the code!!!
    # clustering.hierarchical.hierarchical_clustering(readings)
    # clustering.hierarchical.hierarchical_clustering(data_transform, False)
    # plt.show()

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
    # Moving on to the Hierarchical Clustering on the z-normalised time series data, a different scenario is observed.
    # Again, two main clusters are identified. On the right side of the dendrogram (represented in red) three main
    # clusters appear to form. Further divisions can only be considered for significantly smaller distances, for which
    # a high number of clusters needs to be considered.
    # Analysing the left side of the dendrogram (represented in green) several divisions can be identified. For
    # distances smaller than about 3 a high number of divisions can be identified, and therefore the number of clusters
    # at this distance will be considered as a reference in the initial exploratory analysis. Such a number is 8
    # In this sense, a number of clusters between 4 and 8 are intended to be explored in the K-Means Clustering
    # algorithm. If the cluster evaluation metrics for these limits suggest an invalid cluster formation, this range may
    # be extended.

    # ====================================== K-Means Clustering ========================================================
    # Similarly to Hierarchical Clustering, two distinct K-Means Clustering algorithm implementations will be carried
    # out: At first, K-Means clustering will be performed over the z-normalised time series data, using the DTW
    # distance metric;
    # In a second implementation, reduced time series data (after the PCA transformation) will be used along with the
    # Euclidean distance metric.
    # From the Hierarchical Clustering  step, several ranges of values for the number of clusters were defined for both
    # the approaches. The goal is to test K-Means with these different number of clusters and determine which appears to
    # be the optimal number of clusters, or at least the one that produces the clusters with the best scores in the
    # adopted cluster evaluation metrics.
    # A major characteristic of K-Means clustering is that, depending on how the initial cluster centroids are computed,
    # different results in each run of the algorithm (for the same value of the parameter K) may be produced. This is
    # specially true if a random initialisation is performed.
    # As a result of this characteristic, K-Means clustering is usually executed several times for the same value of K
    # and the initial cluster selection that produces better results is chosen. The question that immediately raises
    # from such a statement is how to determine if one cluster selection is better than another?
    # In 2009 Anil Jain published a scientific article reviewing clustering algorithm, namely K-Means Clustering, where
    # the following suggestion was made:
    #
    # "Typically, K-means is run independently for different values of K and the partition that appears the most
    # meaningful to the domain expert is selected. Different initializations can lead to different final clustering
    # because K-means only converges to local minima. One way to overcome the local minima is to run the K-means
    # algorithm, for a given K , with multiple different initial partitions and choose the partition with the smallest
    # squared error."
    # Jain, A.K. Data clustering: 50 years beyond K-means. Pattern Recognition Lett. (2009),
    # doi: 10.1016/j.patrec.2009.09.011
    #
    # Such a suggestion is only admissible in a supervised clustering scenario, where the sum squared error can be
    # computed for the obtained clusters. In an unsupervised scenario, where the correct grouping of the samples is
    # unknown, such metric cannot be computed; however, Jain's proposal can be adapted and another important cluster
    # evaluation metric can be adopted instead of the squared errors. For instance, it is admissible that clusters
    # are sought so that their within-cluster distances are minimised and between-cluster distances maximised. In this
    # sense, a metric that explores exactly these two properties can be applied. That is the case of the silhouette
    # coefficient (In addition K-Means tries to optimise exactly the parameters evaluated in the silhouette coefficient)

    # As a result, the following approach will be followed in the application of the K-Means Clustering algorithm, for
    # both the raw and reduced time series data:
    #
    # 1) Get ranges for the parameter K from the Dendrogram obtained via Hierarchical Clustering
    # 2) For each value of K to consider
    #       3) Run the K-Means Clustering Algorithm for that value of K, with random centroid initialisation, a
    #          number of times (Say, 10)
    #       4) After each run has finished compute the silhouette coefficient for the clusters obtained
    #       5) Save the clusters from the run that produced the highest silhouette coefficient, yielding clusters with
    #          higher between-cluster distances and lower within-cluster distances
    # 6) Finally, for each set of clusters obtained for each value of K compute cluster evaluation metrics to compare
    #    the obtained results
    # 7) If needed, go back to step 1 and change the ranges for the K values

    # A user on stackexchange proposed the following interpretation for the silhouette coefficient:
    # 0.71-1.0   -> A strong structure has been found
    # 0.51-0.70  -> A reasonable structure has been found
    # 0.26-0.50  -> The structure is weak and could be artificial. Try additional methods of data analysis.
    # < 0.25     -> No substantial structure has been found

    k_values = [2, 3, 4, 5, 6, 7, 8]
    number_runs = 10

    # The main problem of this approach: DTW + Average prototype is that it has been claimed to be a bad combination;
    # average prototype is typically applied to non-elastic distance measures such as the Euclidean. When using DTW
    # local search prototype is common, as well as medoid centroid...
    # best_results_dtw = clustering.k_means.tune_kmeans(readings, k_values, number_runs)

    # When using dba it appears that the silhouette coefficient is even worse than with the dtw + average...
    # Let it run over night so we can then process the results!
    best_results_dba = clustering.k_means.tune_kmeans(readings, k_values, number_runs, True, True)

    # best_results_euclidean = clustering.k_means.tune_kmeans(data_transform, k_values, number_runs, False)

    # I've read some papers ("Time Series Clustering: A decade overview") that claim the application of partitioning
    # methods to time series clustering is a challenging and non-trivial issue. Hierarchical clustering appears as a
    # popular alternative

    # What about using the medoid as the centroid???? According to "Time Series Clustering: A decade overview" is very
    # common in time series clustering

    # TODO: Implement Medoid as centroid

    """
    for _ in range(10):
        centroids, assignments = k_means(readings, 4)

        plt.figure()
        for temp in centroids:
            plt.plot(temp)

    plt.show()
    """
