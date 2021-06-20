import time
from multiprocessing import Pool
import numpy as np
from classifiers.decision_tree import gini, entropy, DecisionTree


class RandomForest:

    def __init__(self, n_trees, max_height=2, min_features=None, min_data=None, errfun=None, n_processes=1):
        self.n_trees = n_trees
        self.max_height = max_height
        self.min_data = min_data
        self.min_features = min_features
        self.forest = []
        self.n_classes = None
        self.errfun = errfun

        self.n_processes = n_processes
        self.train_data = None
        self.train_labels = None

    def fit(self, data, labels):
        self.n_classes = np.max(labels) + 1
        self.train_data = data  # cache train data for the async call
        self.train_labels = labels

        if self.min_data is None:  # checking min sample size
            self.min_data = data.shape[0] // 2
        if self.min_features is None:  # checking min features
            self.min_features = data.shape[1] // 2

        # parallel train each tree
        self.forest = Pool(processes=self.n_processes).map(self.fit_a_tree, range(self.n_trees))
        self.train_data = self.train_labels = None  # clean cache

    def query_forest(self, x):
        bag_result = np.zeros(self.n_classes)

        for classifier in self.forest:
            tree = classifier[0]
            features_map = classifier[1]
            curr = x[features_map]
            curr_pred = tree.query_tree(curr)
            bag_result[curr_pred] += 1
        return np.argmax(bag_result)

    def predict(self, data):
        result = []
        for x in data:
            result.append(self.query_forest(x))
        return np.array(result)

    def fit_a_tree(self, seed):
        curr_seed = int(time.time() // (seed + 1))
        np.random.seed(curr_seed)
        data = self.train_data
        labels = self.train_labels
        # random sample of random sample size
        tree_sample_size = np.random.randint(self.min_data, data.shape[0] + 1)
        tree_sample_map = np.random.choice(range(data.shape[0]), size=tree_sample_size, replace=False)
        # random number of random features
        tree_n_features = np.random.randint(self.min_features, data.shape[1])
        tree_features_map = np.random.choice(range(data.shape[1]), size=tree_n_features, replace=False)

        x = data[tree_sample_map, :]
        x = x[:, tree_features_map]
        y = labels[tree_sample_map]

        # random error function (if not set)
        if self.errfun is None:
            function = np.random.choice([gini, entropy], 1)[0]
        else:
            function = self.errfun

        tree = DecisionTree(function, self.max_height)
        tree.fit(x, y)
        return tree, tree_features_map
