import numpy as np


class Kernel:
    def __init__(self, func, theta=None):
        super().__init__()
        self.func = func
        self.theta = theta

    def get_pairwise(self, A, B):
        return np.repeat(A, B.shape[0], axis=0) - np.tile(B, (A.shape[0], 1))

    def calc(self, A, B):
        return np.reshape(self.func.calc(self.get_pairwise(A,B), self.theta), (A.shape[0], B.shape[0]))

    def calc_gradient(self, A, B, theta):
        return np.reshape(self.func.calc_gradient(self.get_pairwise(A, B), theta), (A.shape[0], B.shape[0]))


class Gaussian:
    def calc(D, theta):
        return np.exp(np.sum(np.square(D) * -theta, axis=1))


class SquaredExponential:
    def calc(D, theta):
        return np.exp(-0.5 * np.sum(np.square(D) * np.exp(-2 * theta), axis=1))



