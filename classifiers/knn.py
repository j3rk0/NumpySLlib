import numpy as np


class KNN:

    def __euclidean_distance(self, x, y):
        sum_sq = np.sum(np.square(x - y))
        return np.sqrt(sum_sq)

    def __init__(self, k, mode="classification", method="standard", smoothing=.1):
        self.mode = mode
        self.method = method
        self.smoothing = smoothing
        self.data = None
        self.n_class = 0
        self.labels = None
        self.k = k
        return

    def predict(self, data):
        if self.mode == "classification":
            return self.classify(data)
        else:
            return self.regression(data)

    def regression(self, data):
        res = []
        for x in data:
            neig = self.find_neighbour(x)
            if self.method == "standard":
                res.append(np.mean(self.labels[neig]))
            else:
                w_tot=r_sum=0
                for n in neig:
                    distance = self.__euclidean_distance(self.data[n],x)
                    r_sum += self.labels[n] / (self.smoothing + distance)
                    w_tot += 1 / (self.smoothing + distance)
                res.append(r_sum / w_tot)
        return res

    def classify(self, data):
        res = []
        for x in data:
            neig = self.find_neighbour(x)
            vote = -1
            if self.method == "standard":
                vote = self.weight_standard(neig)
            elif self.method == "inverse":
                vote = self.weight_inverse_dist(x, neig)
            res.append(np.argmax(vote))
        return np.array(res)

    def fit(self, data, labels):
        self.n_class = np.max(labels) + 1
        self.data = data
        self.labels = labels

    def weight_inverse_dist(self, x, neighbours):
        ret = np.zeros(self.n_class)
        for neig_index in neighbours:
            neighbour = self.data[neig_index, :]
            distance = self.__euclidean_distance(neighbour, x)
            ret[self.labels[neig_index]] += 1 / (distance + self.smoothing)
        return ret

    def weight_standard(self, neighbours):
        ret = np.zeros(self.n_class)
        for neighbour in neighbours:
            ret[self.labels[neighbour]] += 1
        return ret

    def find_neighbour(self, x):
        neighbours = np.ones(self.k + 1) * np.inf
        distances = np.ones(self.k + 1) * np.inf

        for i in range(self.data.shape[0]):
            curr_distance = self.__euclidean_distance(x, self.data[i, :])
            if not np.all(distances > 0) or not np.all(distances[:self.k] < curr_distance):
                neighbours[self.k] = i
                distances[self.k] = curr_distance
                mask = np.argsort(distances)
                neighbours = neighbours[mask]
                distances = distances[mask]

        return neighbours[:self.k].astype(int)
