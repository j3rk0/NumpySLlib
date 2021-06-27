import numpy as np
import random as rndm


class KMeans:
    def __init__(self, k, n_iter):
        self.cluster_labels = None
        self.k = k
        self.n_iter = n_iter

    def fit(self, data):
        set_dim = data.shape[0]
        centroids = [None] * self.k

        # inizializza centroidi random
        centroids_indices = rndm.sample(range(0, set_dim + 1), self.k)
        for i in range(self.k):
            centroids[i] = data[centroids_indices[i]]

        # vettore etichette cluster
        self.cluster_labels = np.ones(set_dim)
        self.cluster_labels *= -1

        for iter in range(self.n_iter):

            # definisce cluster apparteneza di ogni elemento
            for i in range(set_dim):
                nearest_centroid = None
                nearest_centroid_distance = None
                for j in range(self.k):
                    distance = np.sqrt(np.sum(np.square(data[i] - centroids[j]))) # euclidean distance
                    if nearest_centroid is None or nearest_centroid_distance > distance:
                        nearest_centroid = j
                        nearest_centroid_distance = distance
                self.cluster_labels[i] = nearest_centroid

            # suddividiamo i dati per cluster
            clust = [None] * self.k
            for i in range(set_dim):
                if clust[int(self.cluster_labels[i])] is None:
                    clust[int(self.cluster_labels[i])] = data[i]
                else:
                    clust[int(self.cluster_labels[i])] = np.vstack((clust[int(self.cluster_labels[i])], data[i]))

            # ricalcola i centroidi
            for i in range(self.k):
                centroids[i] = self._get_centroid(clust[i])

    def get_cluster_labels(self):
        return self.cluster_labels

    def fit_predict(self, X):
        self.fit(X)
        return self.cluster_labels

    def _get_centroid(self, X):
        dim_clust = X.shape[0]  # numero di istanze nel cluster
        num_features = X.shape[1]  # dimensione spazio delle features

        accumulator = np.zeros(num_features)  # somma accumulata dei punti del cluster

        for x in X:
            accumulator = accumulator + x  # accumulo i vettori nel cluster
        return accumulator / dim_clust  # ritorno il punto medio
