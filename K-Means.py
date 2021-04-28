import numpy as np
import random as rndm


def calcola_centroide(X):
    dim_clust = X.shape[0]  # numero di istanze nel cluster
    num_features = X.shape[1]  # dimensione spazio delle features

    accumulator = np.zeros(num_features)  # somma accumulata dei punti del cluster

    for x in X:
        accumulator = accumulator + x  # accumulo i vettori nel cluster
    return accumulator / dim_clust  # ritorno il punto medio


def rss_cluster(X, mu):  # calcola la distanza tra i punti ed il centroide
    tot = 0  # distanza totale
    for x in X:
        tot += np.sum(np.square(np.abs(x - mu)))  # somma per ogni istanza x  (| x - mu |)^2
    return tot


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
        mu = calcola_centroide(c)  # calcola centroide del cluster
        RSS += rss_cluster(c, mu)  # calcola l'rss del cluster

    return RSS  # ritorna la somma di tutte l'rss


def euclidean_distance(x, y):  # distanza euclidea
    sum_sq = np.sum(np.square(x - y))
    return np.sqrt(sum_sq)


def k_means(data, k, n_iter):
    set_dim = data.shape[0]
    centroids = [None] * k

    # inizializza centroidi random
    centroids_indices = rndm.sample(range(0, set_dim + 1), k)
    for i in range(k):
        centroids[i] = data[centroids_indices[i]]

    # vettore etichette cluster
    labels = np.ones(set_dim)
    labels *= -1

    for iter in range(n_iter):

        # definisce cluster apparteneza di ogni elemento
        for i in range(set_dim):
            nearest_centroid = None
            nearest_centroid_distance = None
            for j in range(k):
                distance = euclidean_distance(data[i], centroids[j])
                if nearest_centroid is None or nearest_centroid_distance > distance:
                    nearest_centroid = j
                    nearest_centroid_distance = distance
            labels[i] = nearest_centroid

        # suddividiamo i dati per cluster
        clust = [None] * k
        for i in range(set_dim):
            if clust[int(labels[i])] is None:
                clust[int(labels[i])] = data[i]
            else:
                clust[int(labels[i])] = np.vstack((clust[int(labels[i])], data[i]))

        # ricalcola i centroidi
        for i in range(k):
            centroids[i] = calcola_centroide(clust[i])

    res = np.hstack((data, np.reshape(labels, (set_dim, 1))))
    for c in centroids:
        curr = np.hstack((c, np.array([-1])))  # inserisce i centroidi etichettandoli con -1
        res = np.vstack((res, curr))
    return res
