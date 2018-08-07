import numpy as np
from sklearn.neighbors import KNeighborsRegressor

from pysao.metamodels.metamodel import Metamodel


class KNeighbors(Metamodel):
    def __init__(self, neighbors, weights):
        Metamodel.__init__(self)
        self.model = None
        self.neighbors = neighbors
        self.weights = weights

    def _predict(self, X):
        F = self.model.predict(X)
        return F, np.zeros(X.shape[0])

    def _fit(self, X, F, data):
        neigh = KNeighborsRegressor(n_neighbors=self.neighbors, weights=self.weights)
        neigh.fit(X, F)
        self.model = neigh
        return self

    @staticmethod
    def get_params():
        val = []
        for neighbors in [1,3,5,10]:
            for w in ['uniform', 'distance']:
                val.append({'neighbors': neighbors, 'weights' : w})
        return val
