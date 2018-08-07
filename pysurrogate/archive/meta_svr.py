import numpy as np
from sklearn.svm import SVR

from pysao.metamodels.metamodel import Metamodel


class SVRMetamodel(Metamodel):
    def __init__(self, kernel, C, epsilon):
        Metamodel.__init__(self)
        self.model = None
        self.kernel = kernel
        self.C = C
        self.epsilon = epsilon

    def _predict(self, X):
        F = self.model.predict(X)
        return F, np.zeros(X.shape[0])

    def _fit(self, X, F, data):
        clf = SVR(kernel=self.kernel, C=self.C, epsilon=self.epsilon)
        clf.fit(X, F)
        self.model = clf
        return self

    @staticmethod
    def get_params():
        val = []
        for kernel in ['linear', 'poly', 'rbf', 'sigmoid']:
            for C in np.linspace(0.2, 2.0, num=10):
                val.append({'kernel': kernel, 'C': C, 'epsilon': 0.0001})
        return val
