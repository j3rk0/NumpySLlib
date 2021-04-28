# %% imports
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.datasets import load_iris
from sklearn.metrics import silhouette_score


def rand_index(clust, labels):
    sample_size = clust.shape[0]
    tp = tn = fp = fn = 0

    for i in range(sample_size):
        for j in range(sample_size):
            if clust[i] == clust[j]:
                if labels[i] == labels[j]:
                    tp += 1
                else:
                    fp += 1
            else:
                if labels[i] == labels[j]:
                    fn += 1
                else:
                    tn += 1

    return (tp + tn) / (tp + tn + fp + fn)


def purity_score(clust, labels):
    n_clust = np.max(clust) + 1  # numero di cluster
    n_class = np.max(labels) + 1
    sample_size = clust.shape[0]
    clusters_size = np.zeros(n_clust)
    clusters_labels = np.zeros((n_clust, n_class))

    for k in range(sample_size):
        cluster_index = clust[k]
        class_index = labels[k]
        clusters_size[cluster_index] += 1  # ho la dim di ogni cluster
        clusters_labels[cluster_index, class_index] += 1

    array_purity = 0
    for z in range(n_clust):
        max_sample_for_class = None
        for t in range(n_class):
            if (max_sample_for_class is None) or (max_sample_for_class < clusters_labels[z, t]):
                max_sample_for_class = clusters_labels[z, t]
        array_purity += max_sample_for_class

    return array_purity / sample_size


def euclidean_distance(x, y):
    sum_sq = np.sum(np.square(x - y))
    return np.sqrt(sum_sq)


def coesion(point_x, data, label, index_cluster):  # coesione di un punto
    dim_data = data.shape[0]
    dist_tot = 0
    dim_cluster = 0

    # calcolo della distanza euclidea tra i punti del cluster e point_x
    for i in range(dim_data):
        if label[i] == index_cluster and not np.array_equal(point_x, data[i]):
            dist_tot += euclidean_distance(point_x, data[i])
            dim_cluster += 1

    result = dist_tot / dim_cluster
    return result


def separation(point_x, data, label, index_clusters):
    num_label = np.max(label) + 1
    dim_data = data.shape[0]
    dim_all_clusters = [0] * num_label
    distance_sum = [0] * num_label

    # calcola la distanza tra point_x e gli altri cluster (vede se sono diversi)
    for i in range(dim_data):
        curr_label = label[i]
        if curr_label != index_clusters:  # attraverso la distanza euclidea
            distance_sum[curr_label] += euclidean_distance(point_x, data[i])
            dim_all_clusters[curr_label] += 1

    min_separation = None

    for i in range(num_label):  # prendo la distanza tra point_x e il cluster più vicino
        if i != index_clusters:
            curr_separation = distance_sum[i] / dim_all_clusters[i]
            if min_separation is None or min_separation > curr_separation:
                min_separation = curr_separation

    return min_separation


def point_silhouette(point_x, data, label, index_clusters):
    separation = separation(point_x, data, label, index_clusters)
    coesion = coesion(point_x, data, label, index_clusters)
    silhouette = 0

    # calcolo la formula della silhouette in base a se la coesione è > o <
    # della separazione
    if coesion > separation:
        silhouette = (coesion - separation) / coesion
    else:
        silhouette = (separation - coesion) / separation

    return silhouette


def silhouette(data, label):  # silhouette per ogni punto
    num_tot_point = data.shape[0]
    sum_all_silhouette = 0

    for i in range(num_tot_point):
        sum_all_silhouette += point_silhouette(data[i], data, label, label[i])

    return sum_all_silhouette / num_tot_point