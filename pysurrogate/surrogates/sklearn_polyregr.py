import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

from pysurrogate.surrogate import Surrogate


class PolynomialRegression(Surrogate):
    def __init__(self, n_degree):
        Surrogate.__init__(self)
        self.model = None
        self.n_degree = n_degree

    def _predict(self, X):
        F = self.model.predict(X)
        return F, np.zeros(X.shape[0])

    def _fit(self, X, F):
        clf = make_pipeline(PolynomialFeatures(self.n_degree), Ridge())
        clf.fit(X, F)
        self.model = clf
        return self

    @staticmethod
    def get_params():
        val = []
        for n_degree in [1, 2, 3, 5]:
            val.append({'n_degree': n_degree})
        return val
