import numpy as np


def train_test_split(X, y, n_train):
    assert n_train <= 1
    n_train = int(n_train * X.shape[0])
    indices = np.array(range(X.shape[0]))
    np.random.shuffle(indices)
    x_train = X[indices[0:n_train]]
    y_train = y[indices[0:n_train]]
    x_test = X[indices[n_train:]]
    y_test = y[indices[n_train:]]
    return x_train, y_train, x_test, y_test


def k_fold(X, y, k):
    fold_size = X.shape[0] // k
    rest = X.shape[0] % k
    assert fold_size > 0

    indices = np.array(range(X.shape[0]))
    np.random.shuffle(indices)

    x_folds = []
    y_folds = []
    start = 0
    for i in range(k):
        curr_fs = fold_size + (rest > 0)  # BALANCING FOLD SIZE
        x_folds.append(X[indices[start:start + curr_fs], :])
        y_folds.append(y[indices[start:start + curr_fs]])
        start += curr_fs
        rest -= 1

    return x_folds, y_folds
