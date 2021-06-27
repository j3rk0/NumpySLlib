import numpy as np


def _euclidean_distance(x, y):
    sum_sq = np.sum(np.square(x - y))
    return np.sqrt(sum_sq)


def _calcola_centroide(X):
    dim_clust = X.shape[0]  # numero di istanze nel cluster
    num_features = X.shape[1]  # dimensione spazio delle features

    accumulator = np.zeros(num_features)  # somma accumulata dei punti del cluster

    for x in X:
        accumulator = accumulator + x  # accumulo i vettori nel cluster
    return accumulator / dim_clust  # ritorno il punto medio


def _rss_cluster(X, mu):  # calcola la distanza tra i punti ed il centroide
    tot = 0  # distanza totale
    for x in X:
        tot += np.sum(np.square(np.abs(x - mu)))  # somma per ogni istanza x  (| x - mu |)^2
    return tot


def coesion(point_x, data, label, index_cluster):  # coesione di un punto
    dim_data = data.shape[0]
    dist_tot = 0
    dim_cluster = 0

    # calcolo della distanza euclidea tra i punti del cluster e point_x
    for i in range(dim_data):
        if label[i] == index_cluster and not np.array_equal(point_x, data[i]):
            dist_tot += _euclidean_distance(point_x, data[i])
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
            distance_sum[curr_label] += _euclidean_distance(point_x, data[i])
            dim_all_clusters[curr_label] += 1

    min_separation = None

    for i in range(num_label):  # prendo la distanza tra point_x e il cluster più vicino
        if i != index_clusters:
            curr_separation = distance_sum[i] / dim_all_clusters[i]
            if min_separation is None or min_separation > curr_separation:
                min_separation = curr_separation

    return min_separation


def rss(X, k, labels):
    set_dimension = X.shape[0]  # dimensione del cluster
    clust = [None] * k  # lista dei cluster

    for i in range(set_dimension):  # suddivide i dati nei cluster di appartenenza
        if (clust[int(labels[i])]) is None:
            clust[int(labels[i])] = X[i]
        else:
            clust[int(labels[i])] = np.vstack((clust[int(labels[i])], X[i]))
    RSS = 0
    for c in clust:
        mu = _calcola_centroide(c)  # calcola centroide del cluster
        RSS += _rss_cluster(c, mu)  # calcola l'rss del cluster

    return RSS  # ritorna la somma di tutte l'rss


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


def silhouette(data, label):  # silhouette per ogni punto
    num_tot_point = data.shape[0]
    sum_all_silhouette = 0

    for i in range(num_tot_point):
        sep = separation(data[i], data, label, label[i])
        coe = coesion(data[i], data, label, label[i])
        sum_all_silhouette += np.abs(coe - sep) / np.maximum(coe, sep)

    return sum_all_silhouette / num_tot_point