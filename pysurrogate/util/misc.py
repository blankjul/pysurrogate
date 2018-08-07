from random import shuffle

import numpy as np


# returns only the unique rows from a given matrix X
def unique_rows(X):
    y = np.ascontiguousarray(X).view(np.dtype((np.void, X.dtype.itemsize * X.shape[1])))
    _, idx = np.unique(y, return_index=True)
    return idx


# creates k-folds to be considered
def create_crossvalidation_sets(n_samples, n_folds, randomize=True):
    res = []
    indices = list(range(n_samples))

    if randomize:
        shuffle(indices)

    slices = [indices[i::n_folds] for i in range(n_folds)]

    for i in range(n_folds):
        validation = slices[i]
        training = [item
                    for s in slices if s is not validation
                    for item in s]
        res.append((list(training), list(validation)))

    return res




