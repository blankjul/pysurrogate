import numpy as np

from pysao.ego.basis import Linear
from pysao.metamodels.metamodel import Metamodel


class MyRBF(Metamodel):

    def __init__(self):
        Metamodel.__init__(self)
        self.model = None

    def _predict(self, X):
        return self.model.predict(X), np.zeros(X.shape[0])

    def _fit(self, X, F, data):
        from pysao.ego.myrbf import MyRBF as MyRBFImpl
        self.model = MyRBFImpl()
        self.model.basis = Linear(X.shape[1])
        self.model.fit(X, F, optimize=None)
        return self

    @staticmethod
    def get_params():
        return [{}]
