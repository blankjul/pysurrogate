import numpy as np

from pysao.eim.evolutionary_interpolation import EvolutionaryInterpolationModel
from pysao.metamodels.metamodel import Metamodel


class EIModel(Metamodel):
    def __init__(self):
        Metamodel.__init__(self)
        self.model = None

    def _predict(self, X):
        return self.model.predict(X), np.zeros(X.shape[0])

    def _fit(self, X, F, data):
        self.model = EvolutionaryInterpolationModel(xl=0 * np.ones(X.shape[1]), xu=1 * np.ones(X.shape[1]))
        self.model.fit(X, F)
        self.model.optimize()
        return self

    @staticmethod
    def get_params():
        val = [{}]
        return val
