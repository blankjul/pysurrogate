import numpy as np
import rbf
import rbf.basis
from rbf.interpolate import RBFInterpolant

from pysao.metamodels.metamodel import Metamodel


class RBFGithubMetamodel(Metamodel):
    def __init__(self, basis, order):
        Metamodel.__init__(self)
        self.basis = basis
        self.order = order
        self.model = None

    def _predict(self, X):
        return self.model(X), np.zeros(X.shape[0])

    def _fit(self, X, F, data):
        #print(self.basis, self.order)
        self.model = RBFInterpolant(X, F, penalty=0.001, sigma=np.full(F.shape[0], 0.001)
                                    ,basis=self.basis, order=self.order)
        return self

    @staticmethod
    def get_params():
        val = []
        for basis in [rbf.basis.phs1, rbf.basis.phs2, rbf.basis.phs3, rbf.basis.phs4, rbf.basis.phs5, rbf.basis.phs6,
                      rbf.basis.phs7, rbf.basis.phs8, rbf.basis.mq, rbf.basis.imq, rbf.basis.iq,
                      rbf.basis.ga,rbf.basis.exp,rbf.basis.se, rbf.basis.mat32, rbf.basis.mat52]:
            for order in [1,2,3,5]:
                val.append({'basis': basis, 'order': order})
        return val

