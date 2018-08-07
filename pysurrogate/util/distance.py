import numpy as np
from scipy.special._ufuncs import xlogy

def eucl_distance(A, B, weights=None, xl=None, xu=None):

    # the difference between all pairs
    D = A-B

    # use the weighted differences if provided
    if weights is not None:
        D *= weights

    # calculate the norm of each row
    M = np.linalg.norm(D, axis=1)

    # normalize by the maximum possible distance
    if xl is not None and xu is not None:
        d = xu - xl
        if weights is not None:
            d *= weights

        M /= np.linalg.norm(d)

    return M


def calc_distance_matrix(A, B, func=eucl_distance):
    _A, _B = get_parwise(A, B)
    return np.reshape(func(_A, _B), (A.shape[0], B.shape[0]))


def get_parwise(A, B):
    return np.tile(B, (len(A), 1)), np.repeat(A, len(B), axis=0)
