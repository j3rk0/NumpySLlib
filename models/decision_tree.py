import numpy as np


def mse(labels):
    if len(labels) == 0:
        return 0

    pred = np.mean(labels)
    err = 0
    for l in labels:
        err += np.square(l - pred)
    return err / len(labels)


def gini(labels):
    label, count = np.unique(labels, return_counts=True)
    prob = count / len(labels)
    return 1 - np.sum(np.square(prob))


def entropy(labels):
    label, count = np.unique(labels, return_counts=True)
    prob = count / len(labels)
    prob *= np.log(prob)
    return - np.sum(prob)


class DecisionTree:

    def __init__(self, mode="classification", errfun="gini", max_height=None, min_err=.0):
        self.root = None
        self.mode = mode
        if errfun == "gini":
            self.errfun = gini
        elif errfun == "entropy":
            self.errfun = entropy
        elif errfun == "mse":
            self.errfun = mse
        self.height = 0
        self.max_height = max_height
        self.min_err = min_err
        self.shape = None

    def __findSplit(self, data, labels):
        n_features = data.shape[1]
        sample_size = data.shape[0]

        curr_best_feat = None
        curr_best_tresh = None
        curr_best_err = None
        for i in range(n_features):  # cerco la feature migliore
            curr = data[:, i]

            if curr_best_feat is None:
                curr_best_feat = i

            for j in range(sample_size):  # cerco il treshold migliore
                curr_treshold = curr[j]

                if curr_best_tresh is None:
                    curr_best_tresh = curr_treshold

                left = labels[curr < curr_treshold]
                right = labels[curr >= curr_treshold]
                left_err = self.errfun(left)  # calcolo errore
                right_err = self.errfun(right)
                err = (left_err * len(left) + right_err * len(right)) / (len(left) + len(right))

                if curr_best_err is None:
                    curr_best_err = err
                elif err < curr_best_err:
                    curr_best_err = err
                    curr_best_tresh = curr_treshold
                    curr_best_feat = i

        return curr_best_feat, curr_best_tresh, curr_best_err

    def fit(self, data, labels, level=0):

        self.shape = data.shape[1]
        if np.all(labels == labels[0]):  # if all element are of the same class let's build a leaf
            self.height = max(level, self.height)
            new_node = self.DecNode(True, label=labels[0], err=0)
        elif self.max_height is not None and level > self.max_height:  # reached max height
            self.height=self.max_height
            err = self.errfun(labels)
            if self.mode == "classification":
                res = np.argmax(np.bincount(labels))
            else:
                res = np.mean(labels)
            return self.DecNode(True, label=res, err=err)
        else:
            feat, tresh, err = self.__findSplit(data, labels)  # find the optimal split

            splitmap = data[:, feat] >= tresh  # split data with a splitmap
            left_data = data[~splitmap, :]
            left_labels = labels[~splitmap]
            right_data = data[splitmap, :]
            right_labels = labels[splitmap]

            if self.mode == "classification":
                left_label = np.argmax(np.bincount(left_labels))
                right_label = np.argmax(np.bincount(right_labels))
            else:
                left_label = np.mean(left_labels)
                right_label = np.mean(right_labels)

            if len(left_labels) == 0:  # left partition is empty => build a leaf
                new_node = self.DecNode(True, label=right_label, err=0)
            elif len(right_labels) == 0:  # right partition is empty => build a leaf
                new_node = self.DecNode(True, label=left_label, err=0)
            else:  # build recursivly two node
                new_node = self.DecNode(False, feature=feat, tresh=tresh, err=err)
                new_node.left = self.fit(left_data, left_labels, level + 1)
                new_node.right = self.fit(right_data, right_labels, level + 1)
        if level == 0:
            self.root = new_node
        return new_node

    def print_tree(self, node=None, buff=None):
        if node is None:
            buff = "IF D[" + str(self.root.feature) + "] >=" + str(self.root.tresh)
            self.print_tree(self.root.right, buff)
            buff = "IF D[" + str(self.root.feature) + "] <" + str(self.root.tresh)
            self.print_tree(self.root.left, buff)
        elif not node.is_leaf:
            buff_dx = buff + " AND " + "D[" + str(node.feature) + "] >=" + str(node.tresh)
            self.print_tree(node.right, buff_dx)
            buff_sx = buff + " AND " + "D[" + str(node.feature) + "] <" + str(node.tresh)
            self.print_tree(node.left, buff_sx)
        else:
            print(buff, " THEN class is ", node.label)

    def query_tree(self, x, returnErr=False):
        assert x.shape[0] == self.shape

        curr_node = self.root

        while not curr_node.is_leaf:  # if leaf is reached, return
            if x[curr_node.feature] >= curr_node.tresh:
                curr_node = curr_node.right
            else:
                curr_node = curr_node.left

        if returnErr:
            return curr_node.err
        return curr_node.label

    def predict(self, X):
        labels = []
        for x in X:
            labels.append(self.query_tree(x))
        return np.array(labels)

    class DecNode:
        def __init__(self, is_leaf, label=None, tresh=None, feature=None, err=None):
            self.is_leaf = is_leaf
            self.err = err
            self.left = None
            self.right = None
            if is_leaf:  # if it's a leaf set the label
                self.label = label
            else:  # if it's inner set infos
                self.feature = feature  # feature evalued
                self.tresh = tresh  # treshold
                # self.err = err  # impurity/gini

# %%
