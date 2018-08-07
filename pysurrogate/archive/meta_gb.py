import numpy as np
from sklearn import ensemble

from pysao.metamodels.metamodel import Metamodel


class GradientBoosting(Metamodel):
    def __init__(self):
        Metamodel.__init__(self)

    def _predict(self, X):
        F = self.model.predict(X)
        return F, np.zeros(X.shape[0])

    def _fit(self, X, F, data):
        clf = ensemble.GradientBoostingRegressor(
                 loss='ls', learning_rate=0.05, n_estimators=300,
                 subsample=1.0, criterion='friedman_mse', min_samples_split=2,
                 min_samples_leaf=1, min_weight_fraction_leaf=0.,
                 max_depth=10
        )
        clf.fit(X, F)
        self.model = clf
        return self

    @staticmethod
    def get_params():
        val = [{}]
        return val

