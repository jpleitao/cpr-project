# coding: utf-8

import os
import csv
import matplotlib.pyplot as plt

import clustering.dba
import clustering.evaluation
import clustering.metrics
import clustering.hierarchical
import clustering.k_means
import clustering.utils

__author__ = 'Joaquim Leitão'
__copyright__ = 'Copyright (c) 2017 Joaquim Leitão'
__email__ = 'jocaleitao93@gmail.com'


def evaluate_clusters(data, time_series, centroid_type):
    if time_series is None:
        time_series = True
    elif not isinstance(time_series, bool):
        raise TypeError('Argument <time_series> must be of type <bool>!')

    if not isinstance(centroid_type, clustering.k_means.CentroidType):
        raise TypeError('Argument <centroid_type> must be of type <clustering.k_means.CentroidType>!')

    # Get data type and distance metric
    if time_series:
        data_type = 'raw'
        dist_metric = 'dtw'
    else:
        data_type = 'reduced'
        dist_metric = 'euclidean'

    # Get centroid type
    if centroid_type == clustering.k_means.CentroidType.AVERAGE:
        cent_type = 'Average'
    elif centroid_type == clustering.k_means.CentroidType.DBA:
        cent_type = 'DBA'
    else:
        cent_type = 'Medoid'

    best_results = clustering.k_means.load_best_results(time_series, centroid_type)
    evaluation = list()
    avg_wc_ss_list = list()
    k_values = list()

    for result in best_results:
        print('============ Starting Evaluation for K = ' + str(result.k) + ' ============================')
        print('Silhouette Coefficient = ' + str(result.silhouette_coefficient))

        ch = clustering.metrics.calinski_herabaz_index(result.assignments, data)
        print('Calinski-Herabaz Index = ' + str(ch))

        sse_sum = clustering.metrics.sse(result.assignments, data, result.centroids, time_series)
        print('Sum Squared Errors = ' + str(sse_sum))
        avg_wc_ss = sse_sum / len(data)
        print('Average Within-Cluster Squared Sums = ' + str(avg_wc_ss))

        avg_wc_ss_list.append(avg_wc_ss)
        k_values.append(result.k)

        evaluation.append([data_type, result.k, dist_metric, cent_type, result.silhouette_coefficient, ch, sse_sum,
                           avg_wc_ss])

    # Plot av_wc_ss for Elbow Method
    fig = plt.figure()
    plt.plot(k_values, avg_wc_ss_list)
    plt.title('Elbow Method')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Average within-cluster sum of squares')
    fig.savefig('data/images/elbow_method/' + str(data_type) + '_' + str(dist_metric) + '_' + str(cent_type))
    plt.close(fig)

    # Append to results file
    results_file = os.getcwd() + '/data/metrics_results.csv'
    if os.path.exists(results_file):
        open_type = 'a'
    else:
        open_type = 'w'

    with open(results_file, open_type) as f:
        writer = csv.writer(f, delimiter=';', quotechar='', quoting=csv.QUOTE_NONE)
        if open_type == 'w':
            # Write header
            writer.writerow(['Data Type', 'Number Clusters', 'Distance Metric', 'Centroid', 'Silhouette Coefficient',
                             'Calinski-Herabaz', 'Sum of Squared Errors', 'Average Within-Cluster Sum Squares'])
        writer.writerows(evaluation)


def clustering_run(readings, data_transform, pca, means, stds, scaler):
    """
    Performs clustering of the time series data. The same clustering methods are intended to be applied to two different
    time series representations: On one of them the raw time series data (with Z-normalisation) is intended to be used;
    on the other the PCA dimensionality reduction algorithm was applied to reduce the time series values to a
    dimensionality capable of explaining 80% of the data variance (3 dimensions)
    :param readings: The original time series values, with Z-Normalisation
    :param data_transform: The time series values in the reduced (that is, transformed) dimensionality
    :param pca: The Principal Components object, used to obtain the time series in reduced dimensionality (its purpose
                is to reverse the process)
    :param means: The means used to compute the z-normalisation
    :param stds: The stds used to compute the Z-normalisation
    :param scaler: The scaler object used to normalise the data with the min-max method
    """
    # ==================================== Hierarchical Clustering =====================================================
    # Start by performing hierarchical clustering on the raw, z-normalised, time series data. For such a data
    # representation an appropriate distance metric must be used, which in this case is the Dynamic Time Warping.
    # Mueen and Keogh highlight the importance of combining Z-Normalisation with DTW for Time Series Clustering in a
    # 2016 Conference Presentation:
    #   "Abdullah Mueen, Eamonn J. Keogh: Extracting Optimal Performance from Dynamic Time Warping. KDD 2016: 2129-2130"
    #
    # We then move on to performing the same task on the transformed data (reduced dimensionality)

    clustering.hierarchical.hierarchical_clustering(readings)
    clustering.hierarchical.hierarchical_clustering(data_transform, False)
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

    # Maybe start the analysis on 2 clusters

    # ====================================== K-Means Clustering ========================================================
    # A user on stackexchange proposed the following interpretation for the silhouette coefficient:
    # 0.71-1.0   -> A strong structure has been found
    # 0.51-0.70  -> A reasonable structure has been found
    # 0.26-0.50  -> The structure is weak and could be artificial. Try additional methods of data analysis.
    # < 0.25     -> No substantial structure has been found

    k_values = [2, 3, 4, 5, 6, 7, 8]
    number_runs = 10

    # **************************** DTW + Average ****************************
    # The main problem of this approach: DTW + Average prototype is that it has been claimed to be a bad combination;
    # average prototype is typically applied to non-elastic distance measures such as the Euclidean. When using DTW
    # local search prototype is common, as well as medoid centroid...
    centroid_type = clustering.k_means.CentroidType.AVERAGE
    best_results_dtw = clustering.k_means.tune_kmeans(readings, k_values, number_runs, True, centroid_type)

    # **************************** DTW + DBA ****************************
    centroid_type = clustering.k_means.CentroidType.DBA
    best_results_dba = clustering.k_means.tune_kmeans(readings, k_values, number_runs, True, centroid_type)

    # **************************** DTW + Medoid ****************************
    # According to "Time Series Clustering: A decade overview" is very common in time series clustering
    centroid_type = clustering.k_means.CentroidType.MEDOID
    best_results_medoid = clustering.k_means.tune_kmeans(readings, k_values, number_runs, True, centroid_type)

    # **************************** PCA, Euclidean + AVERAGE ****************************
    centroid_type = clustering.k_means.CentroidType.AVERAGE
    best_results_euclid = clustering.k_means.tune_kmeans(data_transform, k_values, number_runs, False, centroid_type)

    # I've read some papers ("Time Series Clustering: A decade overview") that claim the application of partitioning
    # methods to time series clustering is a challenging and non-trivial issue. Hierarchical clustering appears as a
    # popular alternative

    # ============================================= Evaluate clusters ==================================================
    # The evaluation of the computed clusters can be performed using internal indexes and/or external indexes. The main
    # difference between the two is that external indexes are used to measure the similarity of formed clusters to
    # externally supplied class labels or ground truth - therefore are supervised metrics - while internal indexes
    # measure the goodness of the computed clusters without resorting to any external information - they are, therefore,
    # unsupervised metrics. Since the current project is inserted in a completely unsupervised scenario, internal
    # indexes must be adopted. Furthermore, the use of graphical inspection methods such as the Elbow method will not be
    # the main focus of this evaluation task, as it enables more subjective interpretations of the results.
    #
    # In this sense, the approach followed with respect to the evaluation of the computed clusters comprises two main
    # steps:
    #   * Initially, a series of quantitative internal metrics were computed for the different clusters. These metrics
    #     were then used to perform an initial comparison of the clusters, discarding solutions that revealed signs of
    #     poor cluster structure.
    #
    #   * In a second iteration, the initially selected cluster solutions were studied more deeply, with metrics and
    #     properties of interest to the field of application of this work being computed and investigated. They include:
    #        -> Average water consumptions (or water profile) for each cluster
    #        -> Number of individual days per cluster
    #        -> Percentage of week days and weekend days per cluster
    #        -> Season of the year more representative in each cluster
    #        -> Etc
    #
    # With respect to the quantitative internal metrics applied in the first step, a list of implemented metrics in this
    # project can be seen in the clustering.metrics module (which also contains implemented distance metrics). The
    # metrics implemented at this stage are:
    #        -> Silhouette Coefficient: Implementation from sklearn.metrics.silhouette_score with precomputed distances
    #                                   (varying depending on the distance metric used to compute the clusters). The SC
    #                                   can be computed as:
    #
    #                                        For each datum i, let the silhouette be:
    #                                               s(i) = ( b(i) - a(i) ) / max(a(i), b(i))
    #                                        where a(i) is the average dissimilarity of i with all other data within
    #                                        the same cluster, and
    #                                        b(i) is the lowest average dissimilarity of i to any other cluster.
    #
    #                                   The silhouette coefficient is, thus, the average of the silhouettes of all data,
    #                                   and is defined in the range [-1; 1], where values closer to 1 suggest more dense
    #                                   and well-separated clusters
    #
    #        -> Calinski-Herabaz Index: Implementation from sklean.metrics.calinski_harabaz_score.
    #                                   According to sklearn's this score is defined as ratio between the within-cluster
    #                                   dispersion and the between-cluster dispersion.
    #                                   This metric is best suited for euclidean distances and its sklearn's
    #                                   implementation explores this distance metric.
    #
    #        -> Sum of Squared Errors: In a time series unsupervised clustering scenario, an error is defined as the
    #                                  distance from a datum to the nearest cluster. This is equivalent to say that
    #                                  this metric is the sum of the squared distances between each datum and their
    #                                  respective centroids. The SSE is often defined as a measure of coherence of the
    #                                  computed clusters, where the smaller its value the "better" the computed clusters
    #

    # DTW + Average
    centroid_type = clustering.k_means.CentroidType.AVERAGE
    evaluate_clusters(readings, True, centroid_type)

    # DTW + DBA
    centroid_type = clustering.k_means.CentroidType.DBA
    evaluate_clusters(readings, True, centroid_type)

    # DTW + Medoid
    centroid_type = clustering.k_means.CentroidType.MEDOID
    evaluate_clusters(readings, True, centroid_type)

    # PCA + Euclidean + Average
    centroid_type = clustering.k_means.CentroidType.AVERAGE
    evaluate_clusters(data_transform, False, centroid_type)

    # Starting with the use of DTW for the distance metric and the average method for the centroid computation, the
    # obtained results are not very optimistic: As expected, a large Silhouette Coefficient value is obtained when only
    # two clusters are being considered; however, the value of this metric drops seriously as further clusters are
    # considered: Silhouette Coefficient values under 0.5 are obtained, suggesting a weak structure of the data
    # clusters. Indeed, as stated in "Time Series Clustering: A Decade Review", computing the average of a collection
    # of time series is not a trivial task, and in cases that Dynamic Time Warping is used as distance metric,
    # averaging prototype is avoided. Analysing the variation of the average within-cluster sum of squares for the
    # different values of k (that is, by inspecting the plot of the elbow method), the plot seems to decrease more
    # slowly after k=5. Even though low Silhouette Coefficient values were recorded for such number of clusters, a
    # deeper analysis of the cluster structure for this number of clusters could be interesting to perform.
    #
    # Results obtained with the Bary Center Averaging (DBA) method followed a similar trend with the main difference
    # that a strong structure of the data is still suggested for k=3 clusters. Further raising the number of clusters
    # lead to considerably low Silhouette Coefficient scores - ranging from about 0.32 all the way down to about 0.2.
    # This suggests a weak cluster structure. When analysing the variation of the average within-cluster sum of squares
    # for the different values of k (that is, by inspecting the plot of the elbow method) a steep drop can be seen when
    # changing k from 3 to 4. If the number of clusters is further increased the average within-cluster sum of squares
    # increases, which suggests a decrease in the cluster quality (samples in the same cluster are farther apart -
    # clusters are not as dense). Based on these observations, a more careful analysis of the clusters obtained for
    # k=3 and k=4 could be performed.
    #
    # When the medoid method was instead used to compute the centroids (keeping DTW as the distance metric) the obtained
    # results suggest a weak cluster structure in the data. In other words, the computed clusters do not appear to have
    # adequate cluster properties: Dense and well-separated. Indeed besides scoring low silhouette coefficient scores,
    # even for a small number of clusters like 2, the average within-cluster sum of squares remains constant, which is
    # not a good characteristic/behaviour.
    #
    # When dimensionality reduction techniques are applied (as in the case of the work of Joana Abreu - Citation!)
    # obtained values for the silhouette coefficient remain quite constant as the number of clusters is increased
    # (ranging between about 0.54 and 0.62), which is a behaviour different from the remainder of the situations.
    # Nevertheless, the mentioned silhouette coefficient values are not very high, suggesting only a reasonable cluster
    # structure. By plotting the variation of the average within-cluster sum of squares for the different values of k
    # (that is, by inspecting the plot of the elbow method), a graphical inspection suggests that for k=5 the decrease
    # in the average within-cluster sum of squares is less steep, meaning that k=5 could be a good target number of
    # clusters with this approach. Furthermore, analysing the values for the Sum of Squared Errors, oscillating values
    # for this metric can be identified: initially it drops as the number of clusters increases from 2 to 3. For k=4
    # this metric registers it highest value (suggesting less dense clusters), only to decrease as k continues to grow.
    #
    # Finally, the application of the TADPole clustering method produced, probably, the most unexpected results mostly
    # due to the fact that, according to the Silhouette Coefficient scores computed, an adequate cluster structure could
    # only be achieved for 2 clusters. The code corresponding to the implementation of the TADPole clustering method
    # can be found in the file 'clustering/clustering.R'. For all the other cases (k=3-8) strongly inadequate cluster
    # structures were determined, resulting in negative scores for the mentioned metric. Another metric that supports
    # the inferences suggested by the analysis of the Silhouette Coefficient scores is the Sum of Squared Errors:
    # Indeed, considerably high values were registered for this metric, being only exceeded in the case of the Euclidean
    # Distance and Average prototype (when dimensionality-reduced data was used, via PCA). A similar scenario is
    # observed for the Average Within-Cluster Sum of Squares.
    # Nevertheless, looking at the elbow plot (containing the variation of the average within-cluster sum of squares
    # with the different values of k considered) a sharp decrease in the average within-cluster sum of squares can be
    # identified when increasing the number of clusters from 3 to 4. For higher numbers of clusters the plotted metric
    # does not decrease as fast. Indeed, such a graphical inspection suggests a deeper inspection of the clusters
    # obtained for these two situations (k=3 and k=4).
    #
    # Generally, a decrease in the value of the silhouette coefficient metric was verified as the value of k increased,
    # which is an expected result. Indeed, for k=2 the values of this metric tend to be high across all experiments
    # except when the medoid was used to compute the centroids, which produced very bad results for all k.
    # Performing a deeper and comparative analysis of the remainder metrics is not an easy task; as these are not
    # limited metrics, its values can considerably change with different distance metrics and cluster centroid
    # computation methods being adopted. Within the same combination of methods a comparison of the values of these
    # metrics can be performed:
    #
    # Regarding the Sum of Squared Errors, its value was expected to decrease as the number of clusters increased.
    # This was only not verified for three combinations of methods: DTW + DBA centroid (k=4-6), DTW + Medoid (value
    # remained the same, as already stated) and Euclidean + Average
    #
    # With respect to the Calinski-Herabaz coefficient, even though higher values suggest more dense and well-separated
    # clusters, the increase in this metric was not always supported by an increase in the silhouette coefficient (for
    # example when clustering time series pre-processed with dimensionality reduction techniques: Euclidean + Average;
    # Or when using the medoid to compute the centroids, along with the dtw metric: DTW + medoid; DTW + TADPole)

    # Based on these results, k=2 could be seen as good number of clusters in the sense that it produces well-separated
    # and dense clusters; however two important aspects must be taken into account at this point: First of all, as
    # expected, when only two clusters are considered the errors (distance of each sample to its cluster centroid) are
    # higher. Secondly, in the context of the problem being tackled, empiric knowledge suggests that more than 2 groups
    # of similar consumption profiles can exist. Obviously, this intuition is based on human experience and may be
    # influenced by consumptions of other regions in the city. In this sense (and specially supported by the high sum of
    # squared errors for k=2) a further results analysis will be performed on the following cases:
    #        -> DTW + Average for k=4 and k=5
    #        -> DTW + DBA for k=3 and k=4
    #        -> Euclidean + Average for k=4 and k=5
    #

    # ========================================= Further Cluster Evaluation =============================================
    # DTW + Average
    centroid_type = clustering.k_means.CentroidType.AVERAGE
    clustering.evaluation.further_cluster_evaluation(readings, True, centroid_type, [4, 5], pca, means, stds, scaler)

    # DTW + DBA
    centroid_type = clustering.k_means.CentroidType.DBA
    clustering.evaluation.further_cluster_evaluation(readings, True, centroid_type, [3, 4], pca, means, stds, scaler)

    # Euclidean + Average
    centroid_type = clustering.k_means.CentroidType.AVERAGE
    clustering.evaluation.further_cluster_evaluation(data_transform, False, centroid_type, [4, 5], pca, means, stds,
                                                     scaler)
