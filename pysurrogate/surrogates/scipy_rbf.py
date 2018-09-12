import numpy as np
from scipy.interpolate import Rbf

from pysurrogate.surrogate import Surrogate


class RBF(Surrogate):
    def __init__(self, kernel):
        Surrogate.__init__(self)
        self.kernel = kernel

    def _predict(self, X):
        args = [X[:, i] for i in range(X.shape[1])]
        return self.model(*args), np.zeros(X.shape[0])

    def _fit(self, X, F):
        args = [X[:, i] for i in range(X.shape[1])]
        args.append(F)
        self.model = Rbf(*args, function=self.kernel)

    @staticmethod
    def get_params():
        val = []
        for rbf in ['multiquadric', 'inverse', 'gaussian', 'linear', 'cubic', 'quintic']:
            val.append({'kernel': rbf})
        return val

