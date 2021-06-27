import numpy as np


def sqrderr(res, labels):
    err = 0
    for i in range(len(labels)):
        err += np.absolute(res[i] - labels[i])
    return err / len(labels)
