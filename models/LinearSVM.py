import numpy as np
from multiprocessing import Pool


class BinaryLinearSVM:
    def __init__(self, max_epochs=1000, learn_rate=1):
        self.W = None
        self.max_epochs = max_epochs
        self.learn_rate = learn_rate
        return

    def fit(self, data, labels):
        n_sample, n_features = data.shape
        self.W = np.zeros(n_features + 1)
        # add bias
        data = np.pad(data, ((0, 0), (0, 1)), constant_values=1)
        # hyperplane encoding
        labels[labels == 0] = -1

        ret = []  # summary of training
        sum_err = 0
        for i in range(self.max_epochs):
            mean_err = 0
            reg = 1 / (i + 1)  # regularization decrease with epochs
            for j in range(n_sample):
                delta = 0
                predicted = data[j, :] @ self.W
                if predicted * labels[j] < 1:  # misclassified
                    delta = (labels[j] * data[j, :])
                    mean_err += 1
                self.W += self.learn_rate * (delta - 2 * reg * self.W)
            sum_err += (mean_err / n_sample)
            ret.append([i, (sum_err / (i + 1))])
        return np.array(ret)  # return summary

    def query_svm(self, x):
        return int(x @ self.W > 0)

    def predict(self, data):
        ret = []
        data = np.pad(data, ((0, 0), (0, 1)), constant_values=1)
        for x in data:
            ret.append(self.query_svm(x))
        return np.array(ret)


class LinearSVM:
    def __init__(self, n_class, max_epochs=1000, learn_rate=1, approach="one-all"):
        assert n_class > 2
        self.n_class = n_class
        self.max_epochs = max_epochs
        self.learn_rate = learn_rate
        self.approach = approach
        self.data = None
        self.labels = [None] * n_class
        self.SVMs = []
        return

    def fit(self, data, labels):
        self.data = data
        self.labels = labels
        self.SVMs = Pool(processes=self.n_class).map(self.fit_a_svm, range(self.n_class))
        self.data = None
        self.labels = None

    def fit_a_svm(self, i):
        currlabels = np.array(self.labels == i, dtype=int)
        svm = BinaryLinearSVM(self.max_epochs, self.learn_rate)
        svm.fit(self.data, currlabels)
        return svm

    def predict(self, data):
        pred = []

        for i in range(self.n_class):
            pred.append(self.SVMs[i].predict(data))
        pred = np.array(pred)
        ret = []
        for i in range(data.shape[0]):
            ret.append(np.argmax(pred[:, i]))
        return np.array(ret)
