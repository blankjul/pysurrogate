import numpy as np
from sklearn.svm import SVC

from pysao.metamodels.metamodel import Metamodel


class SVMMetamodel(Metamodel):
    def __init__(self):
        Metamodel.__init__(self)
        self.model = None

    def _predict(self, X):
        F = self.model.predict(X)
        return F, np.zeros(X.shape[0])

    def _fit(self, X, F, data):

        clf = SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
            decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf',
            max_iter=-1, probability=False, random_state=None, shrinking=True,
            tol=0.001, verbose=False)

        clf.fit(X, F)
        self.model = clf
        return self

    @staticmethod
    def get_params():
        return [{}]

