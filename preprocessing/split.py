import numpy as np


def train_test_split(X, y, n_train):
    assert n_train <= 1
    n_train = int(n_train * X.shape[0])
    sample_map = np.array(range(X.shape[0]))
    np.random.shuffle(sample_map)
    x_train = X[sample_map[0:n_train]]
    y_train = y[sample_map[0:n_train]]
    x_test = X[sample_map[n_train:]]
    y_test = y[sample_map[n_train:]]
    return x_train, y_train, x_test, y_test


def k_fold(X, y, k):
    fold_size = X.shape[0] // k
    rest = X.shape[0] % k
    assert fold_size > 0

    fold_map = np.array(range(X.shape[0]))
    np.random.shuffle(fold_map)

    x_folds = []
    y_folds = []
    start = 0
    for i in range(k):
        curr_fs = fold_size + (rest > 0)  # BALANCING FOLD SIZE
        x_folds.append(X[fold_map[start:start + curr_fs], :])
        y_folds.append(y[fold_map[start:start + curr_fs]])
        start += curr_fs
        rest -= 1

    return x_folds, y_folds
